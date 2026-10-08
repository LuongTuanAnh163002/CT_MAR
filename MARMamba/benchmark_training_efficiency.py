"""Benchmark Stage-3 training efficiency for Vanilla MARMamba and MARMamba-Gate.

This intentionally excludes validation, image/checkpoint saving, and logging from
the timed region. Run both modes on the same idle GPU with identical arguments.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model.mamba import MambaFormer
from model.mamba_multiscale_coarse_gate import (
    GatedMultiScaleCoarseRefinedMambaFormer,
    load_baseline_weights,
)
from utils.aapm_dataset import AAPMTrainDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["vanilla", "gate"], required=True)
    p.add_argument("--checkpoint", required=True,
                   help="Stage-2 checkpoint; used as Vanilla weights or Gate backbone init.")
    p.add_argument("--train_data_dir", required=True)
    p.add_argument("--output", required=True, help="Output JSON path.")
    p.add_argument("--crop_size", nargs=2, type=int, default=[512, 512])
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=19)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--measure_steps", type=int, default=1000)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--estimate_steps", nargs="+", type=int, default=[5000, 20000])
    p.add_argument("--hidden_channels", type=int, default=32)
    p.add_argument("--gate_hidden", type=int, default=32)
    p.add_argument("--gate_range", type=float, default=0.5)
    p.add_argument("--delta_beta", type=float, default=0.01)
    p.add_argument("--lambda_scale2", type=float, default=1.0)
    p.add_argument("--lambda_scale4", type=float, default=1.0)
    p.add_argument("--lambda_final", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_state_flexible(model, path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        if all(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}
        else:
            state = {"module." + k: v for k, v in state.items()}
        model.load_state_dict(state, strict=True)


def hub_loss(pred, gt):
    c = 0.03
    return (torch.sqrt((pred - gt).pow(2) + c * c) - c).mean()


def gate_loss(outputs, gt, beta, lambda_scale2, lambda_scale4, lambda_final):
    with torch.no_grad():
        residual = gt - outputs["baseline"]
        target4 = F.avg_pool2d(residual, 4, 4)
        target2 = F.avg_pool2d(residual, 2, 2)
        target4_to_2 = F.interpolate(target4, size=target2.shape[-2:], mode="nearest")
        detail2 = target2 - target4_to_2
    return (
        lambda_scale2 * F.smooth_l1_loss(
            outputs["pred_detail2_low"], detail2, beta=beta)
        + lambda_scale4 * F.smooth_l1_loss(
            outputs["pred_coarse4_low"], target4, beta=beta)
        + lambda_final * F.smooth_l1_loss(
            outputs["refined"], gt, beta=beta)
    )


def build(args, device):
    if args.model == "vanilla":
        net = MambaFormer(in_channels=1)
        load_state_flexible(net, args.checkpoint)
        net.to(device)
        loss_net = lpips.LPIPS(net="vgg", spatial=False).to(device).eval()
        for p in loss_net.parameters():
            p.requires_grad_(False)
    else:
        net = GatedMultiScaleCoarseRefinedMambaFormer(
            in_channels=1, hidden_channels=args.hidden_channels,
            gate_hidden=args.gate_hidden, gate_range=args.gate_range,
        )
        load_baseline_weights(net, args.checkpoint, map_location="cpu")
        net.to(device)
        loss_net = None
    trainable = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)
    return net, loss_net, optimizer, trainable


def next_batch(iterator, loader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return iterator, batch


def one_step(args, net, loss_net, optimizer, inp, gt, scaler, amp_dtype):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
        if args.model == "vanilla":
            pred = net(inp)
            loss = 0.8 * hub_loss(pred, gt) + 0.2 * loss_net(pred, gt).mean()
        else:
            outputs = net(inp, return_aux=True)
            loss = gate_loss(
                outputs, gt, args.delta_beta,
                args.lambda_scale2, args.lambda_scale4, args.lambda_final,
            )
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach())


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this benchmark.")
    seed_all(args.seed)
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True

    dataset = AAPMTrainDataset(args.crop_size, args.train_data_dir, True, True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0, drop_last=True)
    net, loss_net, optimizer, trainable = build(args, device)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    amp_dtype = torch.float16
    iterator = iter(loader)

    # Warm-up: includes first-use kernel setup but is excluded from measurements.
    print(f"\n[{args.model}] Warm-up: {args.warmup_steps} steps")
    for _ in tqdm(
        range(args.warmup_steps),
        desc=f"{args.model} warm-up",
        unit="step",
        dynamic_ncols=True,
    ):
        iterator, (inp, gt, *_) = next_batch(iterator, loader)
        inp = inp.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        one_step(args, net, loss_net, optimizer, inp, gt, scaler, amp_dtype)
    torch.cuda.synchronize()

    repeats = []
    for repeat in range(args.repeats):
        print(
            f"\n[{args.model}] Measurement repeat "
            f"{repeat + 1}/{args.repeats}: {args.measure_steps} steps"
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        step_ms, data_ms, losses = [], [], []
        progress = tqdm(
            range(args.measure_steps),
            desc=f"{args.model} repeat {repeat + 1}/{args.repeats}",
            unit="step",
            dynamic_ncols=True,
        )
        for _ in progress:
            data_start = time.perf_counter()
            iterator, (inp, gt, *_) = next_batch(iterator, loader)
            inp = inp.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            torch.cuda.synchronize()
            data_ms.append((time.perf_counter() - data_start) * 1000.0)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            losses.append(one_step(args, net, loss_net, optimizer, inp, gt, scaler, amp_dtype))
            end.record()
            torch.cuda.synchronize()
            step_ms.append(start.elapsed_time(end))

            completed = len(step_ms)
            if completed == 1 or completed % 20 == 0:
                running_ms = float(np.mean(step_ms))
                progress.set_postfix(
                    step_ms=f"{running_ms:.1f}",
                    img_s=f"{args.batch_size * 1000.0 / running_ms:.2f}",
                    peak_MiB=f"{torch.cuda.max_memory_allocated(device) / 2**20:.0f}",
                    refresh=False,
                )

        repeats.append({
            "repeat": repeat + 1,
            "gpu_step_ms_mean": float(np.mean(step_ms)),
            "gpu_step_ms_std": float(np.std(step_ms, ddof=1)),
            "data_h2d_ms_mean": float(np.mean(data_ms)),
            "throughput_images_s": float(args.batch_size * 1000.0 / np.mean(step_ms)),
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "final_loss": losses[-1],
        })
        print(
            f"Completed repeat {repeat + 1}/{args.repeats}: "
            f"GPU step={repeats[-1]['gpu_step_ms_mean']:.2f} ms, "
            f"throughput={repeats[-1]['throughput_images_s']:.2f} images/s, "
            f"peak allocated={repeats[-1]['peak_allocated_mib']:.2f} MiB"
        )

    means = np.array([x["gpu_step_ms_mean"] for x in repeats])
    data_means = np.array([x["data_h2d_ms_mean"] for x in repeats])
    peak_alloc = np.array([x["peak_allocated_mib"] for x in repeats])
    peak_reserved = np.array([x["peak_reserved_mib"] for x in repeats])
    mean_ms = float(means.mean())
    end_to_end_ms = float((means + data_means).mean())
    result = {
        "model": args.model,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "resolution": args.crop_size,
        "batch_size": args.batch_size,
        "precision": "amp-fp16" if args.amp else "fp32",
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "repeats": args.repeats,
        "total_params": sum(p.numel() for p in net.parameters()),
        "trainable_params": sum(p.numel() for p in trainable),
        "gpu_step_ms_mean_across_repeats": mean_ms,
        "gpu_step_ms_sd_across_repeats": float(means.std(ddof=1)) if len(means) > 1 else 0.0,
        "end_to_end_step_ms_mean": end_to_end_ms,
        "peak_allocated_mib_mean": float(peak_alloc.mean()),
        "peak_allocated_mib_max": float(peak_alloc.max()),
        "peak_reserved_mib_mean": float(peak_reserved.mean()),
        "estimated_training_time": {
            str(n): {
                "gpu_compute_hours": mean_ms * n / 3_600_000.0,
                "end_to_end_hours": end_to_end_ms * n / 3_600_000.0,
            }
            for n in args.estimate_steps
        },
        "repeat_details": repeats,
        "timing_scope": "forward + training loss + backward + optimizer step; excludes data loading, validation, logging, and checkpoint I/O",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
