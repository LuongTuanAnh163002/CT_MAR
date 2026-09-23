"""Unified progressive-training script for MARMamba comparison models.

Supported architectures:
    - marmamba  -> model/mamba.py::MambaFormer
    - marvit    -> model/vit.py::ViTFormer
    - marpvt    -> model/pvt.py::PVTFormer
    - marformer -> model/marformer.py::MARformer

The ``num_steps`` argument is the cumulative target step. For the three-stage
protocol, use 100000 for stage 1, 300000 for stage 2, and 320000 for stage 3.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict, Mapping

import cv2
import lpips
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as tvu
from torch.utils.data import DataLoader

from model.mamba import MambaFormer
from model.marformer import MARformer
from model.pvt import PVTFormer
from model.vit import ViTFormer
from utils.aapm_dataset import AAPMTrainDataset as MARTrainDataset
from utils.aapm_dataset import test_image
from utils.metrics import calculate_psnr, calculate_rmse, calculate_ssim


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Progressive training for MARMamba and comparison models"
    )
    parser.add_argument(
        "-model",
        "--model",
        required=True,
        choices=["marmamba", "marvit", "marpvt", "marformer"],
        help="Model architecture to train.",
    )
    parser.add_argument("-learning_rate", default=2e-4, type=float)
    parser.add_argument("-crop_size", default=[128, 128], nargs=2, type=int)
    parser.add_argument("-train_batch_size", default=18, type=int)
    parser.add_argument("-val_batch_size", default=1, type=int)
    parser.add_argument("-exp_name", required=True, type=str)
    parser.add_argument("-seed", default=19, type=int)
    parser.add_argument(
        "-num_steps",
        default=90000,
        type=int,
        help="Cumulative target step, not the number of additional steps.",
    )
    parser.add_argument("-checkpoint", default=None, type=str)
    parser.add_argument("-save_step", default=1000, type=int)
    parser.add_argument("-train_data_dir", required=True, type=str)
    parser.add_argument("-val_data_dir", required=True, type=str)
    parser.add_argument("-warm_up", default=False, type=str2bool)
    parser.add_argument("-Tmax", default=10000, type=int)
    parser.add_argument("-num_workers", default=8, type=int)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_model(model_name: str, crop_size):
    input_size = int(crop_size[0])

    if model_name == "marmamba":
        return MambaFormer(in_channels=1)

    if model_name == "marvit":
        return ViTFormer(
            input_size=input_size,
            in_channels=1,
            depth=[1, 2, 2, 4, 1],
            num_heads=[4, 6, 6, 8],
            dim=12,
            bias=True,
            qkv_bias=True,
        )

    if model_name == "marpvt":
        return PVTFormer(
            input_size=input_size,
            in_channels=1,
            depth=[1, 2, 2, 4, 1],
            num_heads=[4, 6, 6, 8],
            sra_size=[7, 7, 7, 7],
            dim=12,
            bias=True,
            qkv_bias=True,
        )

    if model_name == "marformer":
        return MARformer(
            dim=48,
            depth=[1, 2, 3, 4],
            num_heads=[1, 2, 4, 8],
            in_channels=1,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def extract_state_dict(checkpoint) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(checkpoint, dict) and all(
        isinstance(key, str) for key in checkpoint.keys()
    ):
        return checkpoint
    raise TypeError("Checkpoint does not contain a valid model state_dict.")


def strip_module_prefix(state_dict: Mapping[str, torch.Tensor]):
    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def infer_step(checkpoint, checkpoint_path: str) -> int:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("step"), int):
        return checkpoint["step"]

    filename = Path(checkpoint_path).name
    prefix = filename.split("_", maxsplit=1)[0]
    if prefix.isdigit():
        return int(prefix)

    raise ValueError(
        "Cannot infer the training step from checkpoint. Use a filename such "
        "as '100000_ckpt', or store an integer 'step' in the checkpoint."
    )


def load_checkpoint(model: nn.Module, checkpoint_path: str, device) -> int:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = strip_module_prefix(extract_state_dict(checkpoint))
    unwrap_model(model).load_state_dict(state_dict, strict=True)
    step = infer_step(checkpoint, checkpoint_path)
    print(f"--- Loaded checkpoint: {checkpoint_path} (step {step}) ---")
    return step


def save_checkpoint(model: nn.Module, directory: Path, step: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    final_path = directory / f"{step}_ckpt"
    temporary_path = directory / f".{step}_ckpt.tmp"

    # Save an unwrapped state_dict so the file works with both one and multiple GPUs.
    torch.save(unwrap_model(model).state_dict(), temporary_path)
    os.replace(temporary_path, final_path)
    return final_path


def save_image(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tvu.save_image(image.detach().cpu(), str(path))


def huber_like_loss(image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    c = 0.03
    difference = torch.sqrt((image - target) ** 2 + c**2)
    return (difference - c).mean()


def evaluate_saved_images(results_path: Path, gt_path: Path) -> Dict[str, float]:
    image_names = sorted(path.name for path in results_path.iterdir() if path.is_file())
    gt_names = sorted(path.name for path in gt_path.iterdir() if path.is_file())

    if not image_names:
        raise RuntimeError(f"No evaluated images found in {results_path}")
    if len(image_names) != len(gt_names):
        raise RuntimeError(
            f"Output/GT count mismatch: {len(image_names)} vs {len(gt_names)}"
        )

    cumulative_psnr = 0.0
    cumulative_ssim = 0.0
    cumulative_rmse = 0.0

    for image_name, gt_name in zip(image_names, gt_names):
        result = cv2.imread(str(results_path / image_name), cv2.IMREAD_COLOR)
        target = cv2.imread(str(gt_path / gt_name), cv2.IMREAD_COLOR)
        if result is None or target is None:
            raise RuntimeError(f"Failed to read evaluation pair: {image_name}, {gt_name}")

        cumulative_psnr += calculate_psnr(result, target, test_y_channel=True)
        cumulative_ssim += calculate_ssim(result, target, test_y_channel=True)
        cumulative_rmse += calculate_rmse(result, target)

    count = len(image_names)
    return {
        "psnr": cumulative_psnr / count,
        "ssim": cumulative_ssim / count,
        "rmse": cumulative_rmse / count,
    }


def append_metrics(metrics_path: Path, step: int, metrics: Dict[str, float]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as file:
        file.write(
            f"steps:{step}, PSNR:{metrics['psnr']:.6f}, "
            f"SSIM:{metrics['ssim']:.6f}, RMSE:{metrics['rmse']:.6f}\n"
        )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.crop_size[0] <= 0 or args.crop_size[1] <= 0:
        raise ValueError("crop_size values must be positive.")
    if args.num_steps <= 0 or args.save_step <= 0:
        raise ValueError("num_steps and save_step must be positive.")

    experiment_dir = Path(args.exp_name)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = experiment_dir / "eva.txt"
    preview_dir = experiment_dir / "train_res"

    print("--- Hyper-parameters for training ---")
    print(f"model: {args.model}")
    print(f"learning_rate: {args.learning_rate}")
    print(f"crop_size: {args.crop_size}")
    print(f"train_batch_size: {args.train_batch_size}")
    print(f"val_batch_size: {args.val_batch_size}")
    print(f"target cumulative step: {args.num_steps}")
    print(f"training dataset: {args.train_data_dir}")
    print(f"validation dataset: {args.val_data_dir}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, args.crop_size).to(device)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Number of parameters: {parameter_count / 1e6:.4f} M")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    total_steps = 0
    if args.checkpoint:
        total_steps = load_checkpoint(model, args.checkpoint, device)

    if total_steps >= args.num_steps:
        print(
            f"Checkpoint step {total_steps} already reached target "
            f"{args.num_steps}. Nothing to train."
        )
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = None
    if args.warm_up:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.Tmax,
            eta_min=1e-8,
        )
        print(f"Using CosineAnnealingLR with T_max={args.Tmax}")

    perceptual_loss = lpips.LPIPS(net="vgg", spatial=False).to(device)
    perceptual_loss.eval()
    for parameter in perceptual_loss.parameters():
        parameter.requires_grad_(False)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    training_dataset = MARTrainDataset(
        args.crop_size,
        args.train_data_dir,
        random_flip=True,
        random_rotate=True,
    )
    training_loader = DataLoader(
        training_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=args.num_workers > 0,
    )

    # test_image() from the original repository writes to these directories.
    results_path = Path("./eva/output")
    gt_path = Path("./eva/gt")

    while total_steps < args.num_steps:
        for input_image, ground_truth in training_loader:
            if total_steps >= args.num_steps:
                break

            input_image = input_image.to(device, non_blocking=True)
            ground_truth = ground_truth.to(device, non_blocking=True)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(input_image)

            loss = 0.8 * huber_like_loss(prediction, ground_truth)
            loss = loss + 0.2 * perceptual_loss(prediction, ground_truth).mean()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_steps += 1

            if total_steps % 10 == 0:
                print(f"Steps: {total_steps}, loss: {loss.item():.8f}")

            if total_steps % 100 == 0:
                if scheduler is not None:
                    print(f"Current learning rate: {scheduler.get_last_lr()[0]:.10f}")
                save_image(prediction, preview_dir / "output.png")
                save_image(ground_truth, preview_dir / "gt.png")
                save_image(input_image * 0.5 + 0.5, preview_dir / "input.png")

            should_evaluate = (
                total_steps % args.save_step == 0
                or total_steps == args.num_steps
            )
            if should_evaluate:
                model.eval()
                with torch.no_grad():
                    _, average_time = test_image(args.val_data_dir, model)

                print(f"Test speed: {average_time} per image")
                metrics = evaluate_saved_images(results_path, gt_path)
                print(
                    "Validation set, "
                    f"PSNR: {metrics['psnr']:.4f}, "
                    f"SSIM: {metrics['ssim']:.4f}, "
                    f"RMSE: {metrics['rmse']:.4f}"
                )
                append_metrics(metrics_path, total_steps, metrics)
                checkpoint_path = save_checkpoint(
                    model,
                    experiment_dir,
                    total_steps,
                )
                print(f"Saved checkpoint: {checkpoint_path}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"Finish at step {total_steps}!")


if __name__ == "__main__":
    main()
