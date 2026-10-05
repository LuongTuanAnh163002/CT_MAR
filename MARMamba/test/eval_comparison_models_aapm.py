"""Evaluate MARformer, MARViT, or MARPVT on the AAPM CT-MAR split.

This script deliberately follows the same protocol as eval_aapm.py used for
the vanilla MARMamba baseline:

* raw HU values are clipped to [-1000, 3000] and mapped to [0, 1];
* network input is mapped from [0, 1] to [-1, 1];
* predictions are clipped to [0, 1];
* PSNR/SSIM/RMSE are computed on uint8 BGR images;
* LPIPS-VGG receives RGB tensors in [-1, 1];
* metrics are reported both outside the metal mask and with metal included;
* results are stratified by mask-size quartile, single/multiple metal objects,
  and the exact number of metal objects (1--5).

The model constructors match train_step_multi_model.py.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Mapping

import cv2
import lpips
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = (
    SCRIPT_DIR
    if (SCRIPT_DIR / "model").is_dir()
    else SCRIPT_DIR.parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.marformer import MARformer
from model.pvt import PVTFormer
from model.vit import ViTFormer
from utils.metrics import calculate_psnr, calculate_rmse, calculate_ssim


HU_MIN = -1000.0
HU_MAX = 3000.0
FNAME_ID_RE = re.compile(r"img(\d+)")
DIMS_RE = re.compile(r"(\d+)x(\d+)x(\d+)")
REGIONS = ("non_metal", "metal_included")
METRICS = ("psnr", "ssim", "rmse", "lpips")
SIZE_GROUPS = ("Large", "Medium", "Small", "Tiny")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MARformer/MARViT/MARPVT with the MARMamba protocol."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=("marformer", "marvit", "marpvt"),
    )
    parser.add_argument("--test_data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_size", default=512, type=int)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def build_model(model_name: str, input_size: int) -> nn.Module:
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


def hu_to_unit(image: np.ndarray):
    image = np.clip(image, HU_MIN, HU_MAX)
    return ((image - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def find_test_samples(test_data_dir: str):
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
                if match is None:
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
            if match is None:
                continue
            image_id = match.group(1)
            if image_id not in target_map:
                continue
            samples.append(
                {
                    "id": f"{anatomy}_{image_id}",
                    "anatomy": anatomy,
                    "baseline": baseline_path,
                    "target": target_map[image_id],
                    "mask": mask_map.get(image_id),
                    "n_materials": metal_count_map.get(image_id),
                }
            )
        return samples

    if os.path.isdir(os.path.join(test_data_dir, "Baseline")):
        samples = index_anatomy(test_data_dir)
    else:
        samples = []
        for entry in sorted(os.listdir(test_data_dir)):
            anatomy_dir = os.path.join(test_data_dir, entry)
            if os.path.isdir(os.path.join(anatomy_dir, "Baseline")):
                samples.extend(index_anatomy(anatomy_dir))
    return sorted(samples, key=lambda sample: sample["id"])


def size_to_group(size: int, q25: float, q50: float, q75: float):
    if size >= q75:
        return "Large"
    if size >= q50:
        return "Medium"
    if size >= q25:
        return "Small"
    return "Tiny"


def to_uint8_bgr3(image_float01: np.ndarray):
    image_u8 = np.clip(image_float01 * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)


def to_lpips_tensor(image_bgr_uint8: np.ndarray, device):
    image_rgb = cv2.cvtColor(image_bgr_uint8, cv2.COLOR_BGR2RGB)
    image = image_rgb.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)


def extract_state_dict(checkpoint) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint does not contain a valid state_dict.")

    state_dict = {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in checkpoint.items()
        if isinstance(key, str)
    }
    if not state_dict:
        raise TypeError("Checkpoint state_dict is empty.")
    return state_dict


def load_model(model_name: str, input_size: int, checkpoint_path: str, device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = build_model(model_name, input_size).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def evaluate_one(prediction, target, mask, lpips_fn, device, exclude_metal):
    prediction_eval = prediction.copy()
    target_eval = target.copy()
    if exclude_metal and mask is not None:
        prediction_eval[mask] = 0.0
        target_eval[mask] = 0.0

    prediction_bgr = to_uint8_bgr3(prediction_eval)
    target_bgr = to_uint8_bgr3(target_eval)
    prediction_lpips = to_lpips_tensor(prediction_bgr, device)
    target_lpips = to_lpips_tensor(target_bgr, device)
    return {
        "psnr": float(
            calculate_psnr(prediction_bgr, target_bgr, test_y_channel=True)
        ),
        "ssim": float(
            calculate_ssim(prediction_bgr, target_bgr, test_y_channel=True)
        ),
        "rmse": float(calculate_rmse(prediction_bgr, target_bgr)),
        "lpips": float(lpips_fn(prediction_lpips, target_lpips).item()),
    }


def summarize(entries):
    if not entries:
        return None
    arrays = {
        metric: np.asarray([entry[metric] for entry in entries], dtype=np.float64)
        for metric in METRICS
    }
    return {
        "n": len(entries),
        "psnr": f"{arrays['psnr'].mean():.2f} ± {arrays['psnr'].std():.2f}",
        "ssim": f"{arrays['ssim'].mean():.4f} ± {arrays['ssim'].std():.4f}",
        "rmse": f"{arrays['rmse'].mean():.2f} ± {arrays['rmse'].std():.2f}",
        "lpips": f"{arrays['lpips'].mean():.4f} ± {arrays['lpips'].std():.4f}",
        "numeric": {
            metric: {
                "mean": float(values.mean()),
                "std": float(values.std()),
            }
            for metric, values in arrays.items()
        },
    }


def write_per_image_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary):
    print("\n=== OVERALL: all samples ===")
    for region, label in (
        ("non_metal", "Non-metallic area"),
        ("metal_included", "Metal-included region"),
    ):
        result = summary["overall"][region]
        print(
            f"  {label} (n={result['n']}): PSNR={result['psnr']} "
            f"SSIM={result['ssim']} RMSE={result['rmse']} "
            f"LPIPS={result['lpips']}"
        )

    print("\n=== BY METAL-MASK SIZE ===")
    for group in SIZE_GROUPS:
        print(f"\n-- {group} --")
        for region in REGIONS:
            result = summary["by_size"][group][region]
            label = "Non-metal" if region == "non_metal" else "Metal included"
            if result is None:
                print(f"  {label}: no samples")
            else:
                print(
                    f"  {label} (n={result['n']}): PSNR={result['psnr']} "
                    f"SSIM={result['ssim']} RMSE={result['rmse']} "
                    f"LPIPS={result['lpips']}"
                )

    print("\n=== BY MULTIPLICITY ===")
    for group in ("single", "multi"):
        print(f"\n-- {group} --")
        for region in REGIONS:
            result = summary["by_multiplicity"][group][region]
            print(
                f"  {region} (n={result['n']}): PSNR={result['psnr']} "
                f"SSIM={result['ssim']} RMSE={result['rmse']} "
                f"LPIPS={result['lpips']}"
            )

    print("\n=== BY EXACT METAL COUNT ===")
    for count in range(1, 6):
        print(f"\n-- {count} metal --")
        for region in REGIONS:
            result = summary["by_metal_count"][str(count)][region]
            if result is None:
                print(f"  {region}: no samples")
            else:
                print(
                    f"  {region} (n={result['n']}): PSNR={result['psnr']} "
                    f"SSIM={result['ssim']} RMSE={result['rmse']} "
                    f"LPIPS={result['lpips']}"
                )


def run_eval(
    model_name: str,
    test_data_dir: str,
    checkpoint_path: str,
    output_dir: str,
    input_size: int,
    device,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = find_test_samples(test_data_dir)
    if not samples:
        raise RuntimeError(f"No AAPM samples found in {test_data_dir}")

    for sample in samples:
        if sample["n_materials"] is None:
            raise RuntimeError(f"Missing n_materials for {sample['id']}")
        sample["n_materials"] = int(sample["n_materials"])

        if sample["mask"] is None:
            sample["mask_size"] = 0
        else:
            mask = load_raw(sample["mask"], dtype=np.float32) > 0.5
            sample["mask_size"] = int(mask.sum())

    mask_sizes = np.asarray([sample["mask_size"] for sample in samples])
    q25, q50, q75 = np.percentile(mask_sizes, [25, 50, 75])
    for sample in samples:
        sample["size_group"] = size_to_group(
            sample["mask_size"], q25, q50, q75
        )
        sample["multiplicity"] = (
            "multi" if sample["n_materials"] >= 2 else "single"
        )

    print(f"Model: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Input size: {input_size}x{input_size}")
    print(f"Total samples: {len(samples)}")
    print(f"Mask quartiles: Q25={q25:.0f}, Q50={q50:.0f}, Q75={q75:.0f}")

    model = load_model(
        model_name, input_size, checkpoint_path, device
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Parameters: {parameter_count / 1e6:.4f} M")

    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    for parameter in lpips_fn.parameters():
        parameter.requires_grad_(False)

    overall = {region: [] for region in REGIONS}
    by_size = {
        group: {region: [] for region in REGIONS} for group in SIZE_GROUPS
    }
    by_multiplicity = {
        group: {region: [] for region in REGIONS}
        for group in ("single", "multi")
    }
    by_metal_count = {
        count: {region: [] for region in REGIONS} for count in range(1, 6)
    }
    per_image_rows = []
    per_image_details = []

    with torch.inference_mode():
        for sample in tqdm(samples, desc=f"Evaluating {model_name}", unit="image"):
            input_image = hu_to_unit(load_raw(sample["baseline"]))
            target = hu_to_unit(load_raw(sample["target"]))

            if input_image.ndim != 2 or target.ndim != 2:
                raise ValueError(
                    f"Expected 2-D images for {sample['id']}, got "
                    f"{input_image.shape} and {target.shape}"
                )
            if input_image.shape != (input_size, input_size):
                raise ValueError(
                    f"{sample['id']} has shape {input_image.shape}, but "
                    f"--input_size={input_size}"
                )

            mask = None
            if sample["mask"] is not None:
                mask = load_raw(sample["mask"], dtype=np.float32) > 0.5
                if mask.shape != input_image.shape:
                    raise ValueError(
                        f"Mask shape mismatch for {sample['id']}: "
                        f"{mask.shape} vs {input_image.shape}"
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
                for key in ("pred", "prediction", "output"):
                    if key in prediction_tensor:
                        prediction_tensor = prediction_tensor[key]
                        break
                else:
                    raise TypeError("Model output dict has no known prediction key.")

            prediction = np.clip(
                prediction_tensor.squeeze().detach().cpu().numpy(), 0.0, 1.0
            )

            detail = {
                "id": sample["id"],
                "anatomy": sample["anatomy"],
                "n_materials": sample["n_materials"],
                "mask_size": sample["mask_size"],
                "size_group": sample["size_group"],
                "multiplicity": sample["multiplicity"],
            }
            flat_row = {
                "id": sample["id"],
                "anatomy": sample["anatomy"],
                "n_materials": sample["n_materials"],
                "mask_size": sample["mask_size"],
                "size_group": sample["size_group"],
                "multiplicity": sample["multiplicity"],
            }

            for exclude_metal, region in (
                (True, "non_metal"),
                (False, "metal_included"),
            ):
                metrics = evaluate_one(
                    prediction,
                    target,
                    mask,
                    lpips_fn,
                    device,
                    exclude_metal,
                )
                entry = {"id": sample["id"], **metrics}
                overall[region].append(entry)
                by_size[sample["size_group"]][region].append(entry)
                by_multiplicity[sample["multiplicity"]][region].append(entry)
                if sample["n_materials"] in by_metal_count:
                    by_metal_count[sample["n_materials"]][region].append(entry)

                detail[region] = metrics
                for metric, value in metrics.items():
                    flat_row[f"{region}_{metric}"] = value

            per_image_details.append(detail)
            per_image_rows.append(flat_row)

    summary = {
        "metadata": {
            "model": model_name,
            "checkpoint": checkpoint_path,
            "input_size": input_size,
            "n_samples": len(samples),
            "hu_range": [HU_MIN, HU_MAX],
            "mask_size_quartiles": {
                "q25": float(q25),
                "q50": float(q50),
                "q75": float(q75),
            },
            "protocol": (
                "Same uint8-BGR PSNR/SSIM/RMSE and LPIPS-VGG protocol "
                "as vanilla MARMamba eval_aapm.py"
            ),
        },
        "overall": {
            region: summarize(overall[region]) for region in REGIONS
        },
        "by_size": {
            group: {
                region: summarize(by_size[group][region]) for region in REGIONS
            }
            for group in SIZE_GROUPS
        },
        "by_multiplicity": {
            group: {
                region: summarize(by_multiplicity[group][region])
                for region in REGIONS
            }
            for group in ("single", "multi")
        },
        "by_metal_count": {
            str(count): {
                region: summarize(by_metal_count[count][region])
                for region in REGIONS
            }
            for count in range(1, 6)
        },
    }

    with open(output_dir / "eval_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    with open(
        output_dir / "per_image_metrics.json", "w", encoding="utf-8"
    ) as file:
        json.dump(per_image_details, file, indent=2, ensure_ascii=False)
    write_per_image_csv(
        per_image_rows, output_dir / "per_image_metrics.csv"
    )

    print_summary(summary)
    print(f"\nSaved summary: {output_dir / 'eval_summary.json'}")
    print(f"Saved per-image JSON: {output_dir / 'per_image_metrics.json'}")
    print(f"Saved per-image CSV: {output_dir / 'per_image_metrics.csv'}")
    return summary


def main():
    args = parse_args()
    if args.input_size <= 0:
        raise ValueError("--input_size must be positive.")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    output_dir = args.output_dir or f"./eval_{args.model}_{args.input_size}"
    run_eval(
        model_name=args.model,
        test_data_dir=args.test_data_dir,
        checkpoint_path=args.checkpoint,
        output_dir=output_dir,
        input_size=args.input_size,
        device=device,
    )


if __name__ == "__main__":
    main()
