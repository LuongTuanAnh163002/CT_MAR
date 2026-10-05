"""Select the best AAPM checkpoint for MARMamba comparison models.

Supported models:
    marmamba, marformer, marvit, marpvt

Selection rule:
    highest overall non-metal validation PSNR

The preprocessing and metric protocol matches the MARMamba/Gate evaluation:
    raw HU -> [0, 1]
    network input -> [-1, 1]
    prediction clipped to [0, 1]
    prediction and GT are both set to zero inside the metal mask
    PSNR/SSIM/RMSE are computed on uint8 BGR images
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.mamba import MambaFormer
from model.marformer import MARformer
from model.pvt import PVTFormer
from model.vit import ViTFormer
from utils.metrics import calculate_psnr, calculate_rmse, calculate_ssim


HU_MIN = -1000.0
HU_MAX = 3000.0
FNAME_ID_RE = re.compile(r"img(\d+)")
DIMS_RE = re.compile(r"(\d+)x(\d+)x(\d+)")
STEP_RE = re.compile(r"^(\d+)_ckpt(?:\.(?:pt|pth))?$")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select the best checkpoint by overall non-metal validation PSNR."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["marmamba", "marformer", "marvit", "marpvt"],
        help="Architecture used to create the checkpoints.",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        required=True,
        help="Evaluation resolution and ViT/PVT constructor input size, e.g. 512.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Directory containing checkpoints such as 301000_ckpt.",
    )
    parser.add_argument(
        "--val_data_dir",
        required=True,
        help="AAPM validation root containing body*/head* directories.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Default: ./<model>_<input_size>_checkpoint_selection",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start_step", type=int, default=None)
    parser.add_argument("--end_step", type=int, default=None)
    parser.add_argument(
        "--step_interval",
        type=int,
        default=None,
        help="Optional filter relative to start_step.",
    )
    return parser.parse_args()


def build_model(model_name: str, input_size: int) -> nn.Module:
    if model_name == "marmamba":
        return MambaFormer(in_channels=1)

    if model_name == "marformer":
        return MARformer(
            dim=48,
            depth=[1, 2, 3, 4],
            num_heads=[1, 2, 4, 8],
            in_channels=1,
        )

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

    raise ValueError(f"Unsupported model: {model_name}")


def parse_dims(filename: str):
    match = DIMS_RE.search(filename)
    if match is None:
        raise ValueError(f"Cannot parse dimensions from: {filename}")
    width, height, depth = (int(value) for value in match.groups())
    return height, width, depth


def load_raw(path: str, dtype=np.float32):
    rows, cols, slices = parse_dims(os.path.basename(path))
    array = np.fromfile(path, dtype=dtype)
    expected = rows * cols * slices
    if array.size != expected:
        raise ValueError(f"{path}: found {array.size} values, expected {expected}")
    if slices == 1:
        return array.reshape(rows, cols)
    return array.reshape(slices, rows, cols)


def hu_to_unit(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, HU_MIN, HU_MAX)
    return ((image - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def find_eval_samples(root_dir: str):
    def index_anatomy(anatomy_dir: str):
        anatomy = os.path.basename(os.path.normpath(anatomy_dir))
        baseline_dir = os.path.join(anatomy_dir, "Baseline")
        target_dir = os.path.join(anatomy_dir, "Target")
        mask_dir = os.path.join(anatomy_dir, "Mask")

        if not os.path.isdir(baseline_dir) or not os.path.isdir(target_dir):
            return []

        target_map = {}
        for path in glob.glob(os.path.join(target_dir, "*.raw")):
            match = FNAME_ID_RE.search(os.path.basename(path))
            if match:
                target_map[match.group(1)] = path

        mask_map = {}
        metal_count_map = {}
        if os.path.isdir(mask_dir):
            for path in glob.glob(os.path.join(mask_dir, "*.raw")):
                match = FNAME_ID_RE.search(os.path.basename(path))
                if match:
                    mask_map[match.group(1)] = path

            for path in glob.glob(os.path.join(mask_dir, "*.json")):
                match = re.search(r"metalinfo(\d+)", os.path.basename(path))
                if not match:
                    continue
                try:
                    with open(path, "r", encoding="utf-8") as file:
                        metal_count_map[match.group(1)] = json.load(file).get(
                            "n_materials"
                        )
                except (OSError, json.JSONDecodeError):
                    pass

        samples = []
        for baseline_path in glob.glob(os.path.join(baseline_dir, "*.raw")):
            match = FNAME_ID_RE.search(os.path.basename(baseline_path))
            if not match:
                continue
            image_id = match.group(1)
            if image_id not in target_map:
                continue
            samples.append(
                {
                    "id": f"{anatomy}_{image_id}",
                    "baseline": baseline_path,
                    "target": target_map[image_id],
                    "mask": mask_map.get(image_id),
                    "n_materials": metal_count_map.get(image_id),
                }
            )
        return samples

    if os.path.isdir(os.path.join(root_dir, "Baseline")):
        samples = index_anatomy(root_dir)
    else:
        samples = []
        for name in sorted(os.listdir(root_dir)):
            anatomy_dir = os.path.join(root_dir, name)
            if os.path.isdir(anatomy_dir):
                samples.extend(index_anatomy(anatomy_dir))
    return sorted(samples, key=lambda item: item["id"])


def to_uint8_bgr3(image_float01: np.ndarray) -> np.ndarray:
    image_u8 = np.clip(image_float01 * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)


def metric_one_non_metal(prediction, target, mask):
    prediction_eval = prediction.copy()
    target_eval = target.copy()
    if mask is not None:
        prediction_eval[mask] = 0.0
        target_eval[mask] = 0.0

    prediction_bgr = to_uint8_bgr3(prediction_eval)
    target_bgr = to_uint8_bgr3(target_eval)
    return {
        "psnr": float(
            calculate_psnr(prediction_bgr, target_bgr, test_y_channel=True)
        ),
        "ssim": float(
            calculate_ssim(prediction_bgr, target_bgr, test_y_channel=True)
        ),
        "rmse": float(calculate_rmse(prediction_bgr, target_bgr)),
    }


def summarize(records):
    if not records:
        return None
    result = {"n": len(records)}
    for metric in ("psnr", "ssim", "rmse"):
        values = np.asarray(
            [record["metrics"][metric] for record in records], dtype=np.float64
        )
        result[metric] = float(values.mean())
        result[f"{metric}_std"] = float(values.std())
    return result


def find_checkpoints(
    checkpoint_dir: str,
    start_step=None,
    end_step=None,
    step_interval=None,
):
    if not os.path.isdir(checkpoint_dir):
        raise NotADirectoryError(
            f"Checkpoint directory does not exist: {checkpoint_dir}"
        )

    found = []
    ignored_entries = []

    # Scan only direct children. Directories such as train_res and files such
    # as eva.txt are expected in a training output folder and are ignored.
    for entry in os.scandir(checkpoint_dir):
        if not entry.is_file():
            ignored_entries.append(entry.name)
            continue

        match = STEP_RE.fullmatch(entry.name)
        if match is None:
            ignored_entries.append(entry.name)
            continue

        step = int(match.group(1))
        if start_step is not None and step < start_step:
            continue
        if end_step is not None and step > end_step:
            continue
        if step_interval is not None:
            anchor = start_step if start_step is not None else 0
            if (step - anchor) % step_interval != 0:
                continue
        found.append((step, entry.path))

    found.sort(key=lambda item: item[0])
    if not found:
        raise RuntimeError(
            f"No checkpoints found in {checkpoint_dir} for the requested range."
        )

    if ignored_entries:
        preview = ", ".join(sorted(ignored_entries)[:10])
        remainder = len(ignored_entries) - min(10, len(ignored_entries))
        suffix = f" (+{remainder} more)" if remainder else ""
        print(
            "Ignored non-checkpoint entries in checkpoint_dir: "
            f"{preview}{suffix}"
        )

    return found


def extract_state_dict(checkpoint) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint does not contain a state_dict.")

    state_dict = {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in checkpoint.items()
        if isinstance(key, str)
    }
    if not state_dict:
        raise TypeError("Checkpoint state_dict is empty or invalid.")
    return state_dict


def load_checkpoint_into_model(model, checkpoint_path: str, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()


@torch.inference_mode()
def evaluate_checkpoint(model, checkpoint_path, samples, device, input_size):
    load_checkpoint_into_model(model, checkpoint_path, device)
    records = []
    start_time = time.time()

    progress = tqdm(samples, desc="validation", leave=False)
    for sample in progress:
        input_image = hu_to_unit(load_raw(sample["baseline"]))
        target_image = hu_to_unit(load_raw(sample["target"]))

        if input_image.ndim != 2 or target_image.ndim != 2:
            raise ValueError(
                f"Expected one 2-D slice for {sample['id']}, got "
                f"{input_image.shape} and {target_image.shape}."
            )
        if input_image.shape != (input_size, input_size):
            raise ValueError(
                f"{sample['id']} has shape {input_image.shape}, but "
                f"--input_size={input_size}. Use the matching resolution."
            )

        mask = None
        if sample["mask"] is not None:
            mask = load_raw(sample["mask"], dtype=np.float32) > 0.5
            if mask.shape != input_image.shape:
                raise ValueError(
                    f"Mask shape mismatch for {sample['id']}: "
                    f"{mask.shape} vs {input_image.shape}."
                )

        input_tensor = (
            torch.from_numpy((input_image - 0.5) / 0.5)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(device)
        )
        prediction_tensor = model(input_tensor)
        if isinstance(prediction_tensor, (tuple, list)):
            prediction_tensor = prediction_tensor[0]
        if isinstance(prediction_tensor, dict):
            for key in ("pred", "prediction", "output", "refined"):
                if key in prediction_tensor:
                    prediction_tensor = prediction_tensor[key]
                    break
            else:
                raise TypeError("Model returned a dict without a known output key.")

        prediction = np.clip(
            prediction_tensor.squeeze().detach().cpu().numpy(), 0.0, 1.0
        )
        n_materials = int(sample["n_materials"])
        records.append(
            {
                "id": sample["id"],
                "n_materials": n_materials,
                "group": "multi" if n_materials >= 2 else "single",
                "metrics": metric_one_non_metal(
                    prediction, target_image, mask
                ),
            }
        )

    single_records = [r for r in records if r["group"] == "single"]
    multi_records = [r for r in records if r["group"] == "multi"]
    result = {
        "overall": summarize(records),
        "single": summarize(single_records),
        "multi": summarize(multi_records),
        "by_metal_count": {},
        "seconds": float(time.time() - start_time),
    }
    for count in range(1, 6):
        subset = [r for r in records if r["n_materials"] == count]
        result["by_metal_count"][str(count)] = summarize(subset)
    return result


def print_result(step: int, result):
    overall = result["overall"]
    multi = result["multi"]
    multi_psnr = "N/A" if multi is None else f"{multi['psnr']:.6f}"
    print(
        f"step {step:>7} | "
        f"Overall PSNR={overall['psnr']:.6f} "
        f"SSIM={overall['ssim']:.6f} "
        f"RMSE={overall['rmse']:.6f} | "
        f"Multi PSNR={multi_psnr} | "
        f"time={result['seconds']:.1f}s"
    )


def save_json(path: str, payload):
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    os.replace(temporary_path, path)


def main():
    args = parse_args()
    if args.input_size <= 0:
        raise ValueError("--input_size must be positive.")
    if args.step_interval is not None and args.step_interval <= 0:
        raise ValueError("--step_interval must be positive.")
    if args.start_step is not None and args.end_step is not None:
        if args.start_step > args.end_step:
            raise ValueError("--start_step cannot exceed --end_step.")

    output_dir = args.output_dir or (
        f"./{args.model}_{args.input_size}_checkpoint_selection"
    )
    os.makedirs(output_dir, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)

    samples = find_eval_samples(args.val_data_dir)
    if not samples:
        raise RuntimeError(f"No validation samples found in {args.val_data_dir}")

    missing_counts = [s["id"] for s in samples if s["n_materials"] is None]
    if missing_counts:
        raise RuntimeError(
            "Missing n_materials for "
            f"{len(missing_counts)} samples. Examples: {missing_counts[:5]}"
        )

    count_histogram = {count: 0 for count in range(1, 6)}
    for sample in samples:
        count = int(sample["n_materials"])
        if count not in count_histogram:
            raise RuntimeError(f"Unexpected n_materials={count} for {sample['id']}")
        count_histogram[count] += 1

    checkpoints = find_checkpoints(
        args.checkpoint_dir,
        start_step=args.start_step,
        end_step=args.end_step,
        step_interval=args.step_interval,
    )

    print(f"Model: {args.model}")
    print(f"Input size: {args.input_size}x{args.input_size}")
    print(f"Validation samples: {len(samples)}")
    print(f"Metal-count histogram: {count_histogram}")
    print(f"Checkpoints found: {len(checkpoints)}")

    model = build_model(args.model, args.input_size).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model parameters: {parameter_count / 1e6:.4f} M")

    all_results = []
    result_path = os.path.join(output_dir, "checkpoint_results.json")
    for index, (step, checkpoint_path) in enumerate(checkpoints, start=1):
        print(
            f"\n[{index}/{len(checkpoints)}] "
            f"Evaluating step {step}: {checkpoint_path}"
        )
        result = evaluate_checkpoint(
            model,
            checkpoint_path,
            samples,
            device,
            args.input_size,
        )
        print_result(step, result)
        all_results.append(
            {"step": step, "checkpoint": checkpoint_path, **result}
        )
        save_json(
            result_path,
            {
                "model": args.model,
                "input_size": args.input_size,
                "selection_rule": "highest overall non-metal validation PSNR",
                "n_samples": len(samples),
                "metal_count_histogram": {
                    str(key): int(value)
                    for key, value in count_histogram.items()
                },
                "results": all_results,
            },
        )

    ranking = sorted(
        all_results,
        key=lambda item: item["overall"]["psnr"],
        reverse=True,
    )
    best = ranking[0]

    print("\n" + "=" * 100)
    print(
        f"BEST {args.model.upper()} CHECKPOINT "
        "(Overall Non-metal Validation PSNR)"
    )
    print("=" * 100)
    print(f"Step:       {best['step']}")
    print(f"Checkpoint: {best['checkpoint']}")
    print(f"PSNR:       {best['overall']['psnr']:.8f}")
    print(f"SSIM:       {best['overall']['ssim']:.8f}")
    print(f"RMSE:       {best['overall']['rmse']:.8f}")

    print("\nTop checkpoints:")
    for rank, item in enumerate(ranking[: min(10, len(ranking))], start=1):
        print(
            f"{rank:>2}. step={item['step']:>7} | "
            f"PSNR={item['overall']['psnr']:.6f} | "
            f"SSIM={item['overall']['ssim']:.6f} | "
            f"RMSE={item['overall']['rmse']:.6f}"
        )

    ranking_path = os.path.join(output_dir, "checkpoint_ranking.json")
    save_json(
        ranking_path,
        {
            "model": args.model,
            "input_size": args.input_size,
            "selection_rule": "highest overall non-metal validation PSNR",
            "best": best,
            "ranking": ranking,
        },
    )

    best_path = os.path.join(output_dir, "best_checkpoint.txt")
    with open(best_path, "w", encoding="utf-8") as file:
        file.write("selection_rule=highest overall non-metal validation PSNR\n")
        file.write(f"model={args.model}\n")
        file.write(f"input_size={args.input_size}\n")
        file.write(f"step={best['step']}\n")
        file.write(f"checkpoint={best['checkpoint']}\n")
        file.write(f"psnr={best['overall']['psnr']}\n")
        file.write(f"ssim={best['overall']['ssim']}\n")
        file.write(f"rmse={best['overall']['rmse']}\n")

    print(f"\nSaved all results: {result_path}")
    print(f"Saved ranking:     {ranking_path}")
    print(f"Saved best info:   {best_path}")


if __name__ == "__main__":
    main()
