#!/usr/bin/env python3
"""Zero-shot qualitative evaluation of AAPM-trained MARMamba models on CLINIC-metal.

The script:
  1. loads CLINIC-metal NIfTI volumes (.nii or .nii.gz),
  2. selects diverse axial slices with the largest number of voxels > 2500 HU,
  3. applies the exact AAPM normalization used during training,
  4. runs Vanilla MARMamba and MARMamba-Gate,
  5. restores original implant pixels only for clinical visualization,
  6. saves individual images, comparison panels, and metadata.csv.

This is a no-reference qualitative evaluation. It intentionally does not report
PSNR, SSIM, RMSE, or LPIPS because CLINIC-metal has no paired clean target.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from PIL import Image


HU_MIN = -1000.0
HU_MAX = 3000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Vanilla and MARMamba-Gate on CLINIC-metal NIfTI volumes."
    )
    parser.add_argument("--repo_dir", required=True, help="Root of the MARMamba repository.")
    parser.add_argument("--data_path", required=True, help="Folder containing CLINIC-metal NIfTI files.")
    parser.add_argument("--vanilla_checkpoint", required=True, help="Vanilla 512 best checkpoint.")
    parser.add_argument("--gate_checkpoint", required=True, help="Gate .pt train state or raw checkpoint.")
    parser.add_argument("--save_path", default="/kaggle/working/clinic_metal_results")
    parser.add_argument("--top_k_per_volume", default=3, type=int)
    parser.add_argument(
        "--min_slice_gap",
        default=10,
        type=int,
        help="Minimum axial-index distance between selected slices in one volume.",
    )
    parser.add_argument("--metal_threshold", default=2500.0, type=float)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--max_volumes", default=None, type=int)
    parser.add_argument("--seed", default=19, type=int)
    parser.add_argument("--hidden_channels", default=32, type=int)
    parser.add_argument("--gate_hidden", default=32, type=int)
    parser.add_argument("--gate_range", default=0.5, type=float)
    parser.add_argument("--soft_window", default=[-175.0, 275.0], nargs=2, type=float)
    parser.add_argument("--bone_window", default=[-500.0, 1500.0], nargs=2, type=float)
    parser.add_argument("--difference_limit_hu", default=300.0, type=float)
    parser.add_argument(
        "--no_preserve_metal",
        action="store_true",
        help="Do not restore original >threshold implant pixels in saved outputs.",
    )
    parser.add_argument("--save_npz", action="store_true", help="Save HU arrays for each selected slice.")
    parser.add_argument(
        "--panel_only",
        action="store_true",
        help="Save only one combined panel per selected slice; skip individual PNGs.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_nifti_files(data_path: str) -> List[str]:
    patterns = [
        os.path.join(data_path, "**", "*_data.nii"),
        os.path.join(data_path, "**", "*_data.nii.gz"),
    ]
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(
            f"No *_data.nii or *_data.nii.gz files found under: {data_path}"
        )
    return files


def extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint is not a mapping/state_dict.")

    for key in ("model", "state_dict", "model_state_dict", "net"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping) and value:
            checkpoint = value
            break

    if not checkpoint or not all(isinstance(k, str) for k in checkpoint.keys()):
        raise ValueError("Could not locate a model state_dict in checkpoint.")
    return checkpoint


def strip_common_prefixes(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    result = dict(state)
    # DataParallel is the common case. The other prefixes make the loader robust
    # to common training wrappers without changing valid architecture keys.
    for prefix in ("module.", "model.", "net."):
        if result and all(key.startswith(prefix) for key in result):
            result = {key[len(prefix):]: value for key, value in result.items()}
    return result


def load_model_checkpoint(model: torch.nn.Module, path: str, device: torch.device) -> None:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    state = strip_common_prefixes(extract_state_dict(checkpoint))
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        model_keys = list(model.state_dict().keys())[:8]
        checkpoint_keys = list(state.keys())[:8]
        raise RuntimeError(
            f"Failed to load checkpoint strictly: {path}\n"
            f"Model key examples: {model_keys}\n"
            f"Checkpoint key examples: {checkpoint_keys}"
        ) from exc
    model.to(device).eval()


def hu_to_unit(image_hu: np.ndarray) -> np.ndarray:
    image_hu = np.clip(image_hu, HU_MIN, HU_MAX)
    return ((image_hu - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def unit_to_hu(image_unit: np.ndarray) -> np.ndarray:
    return image_unit.astype(np.float32) * (HU_MAX - HU_MIN) + HU_MIN


def window_image(image_hu: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        raise ValueError("Window high must be larger than window low.")
    image = np.clip(image_hu, low, high)
    return ((image - low) / (high - low)).astype(np.float32)


def select_diverse_slices(
    metal_scores: np.ndarray,
    top_k: int,
    min_gap: int,
) -> List[int]:
    if top_k <= 0:
        return []
    ranked = np.argsort(metal_scores)[::-1]
    selected: List[int] = []
    for index in ranked.tolist():
        if metal_scores[index] <= 0:
            break
        if all(abs(index - previous) >= min_gap for previous in selected):
            selected.append(int(index))
        if len(selected) == top_k:
            break
    # A volume may have metal in fewer separated regions than requested.
    if not selected and len(metal_scores):
        selected = [int(np.argmax(metal_scores))]
    return selected


def tensorize_slices(slices_hu: Sequence[np.ndarray], device: torch.device) -> torch.Tensor:
    units = np.stack([hu_to_unit(image) for image in slices_hu], axis=0)
    model_inputs = units * 2.0 - 1.0
    tensor = torch.from_numpy(np.ascontiguousarray(model_inputs)).unsqueeze(1)
    return tensor.float().to(device, non_blocking=True)


def unwrap_gate_output(output: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
    stats: Dict[str, Any] = {}
    if torch.is_tensor(output):
        return output, stats
    if isinstance(output, (tuple, list)):
        if not output or not torch.is_tensor(output[0]):
            raise TypeError("Unexpected Gate tuple output.")
        if len(output) >= 4 and isinstance(output[3], Mapping):
            stats = dict(output[3])
        return output[0], stats
    if isinstance(output, Mapping):
        for key in ("refined", "output", "pred", "prediction"):
            if key in output and torch.is_tensor(output[key]):
                stats_obj = output.get("stats", {})
                if isinstance(stats_obj, Mapping):
                    stats = dict(stats_obj)
                return output[key], stats
    raise TypeError(f"Unsupported Gate output type: {type(output)!r}")


def scalar_for_item(value: Any, index: int) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.item())
        if value.shape[0] > index:
            item = value[index]
            if item.numel() == 1:
                return float(item.item())
            return json.dumps(item.reshape(-1).tolist())
        return json.dumps(value.reshape(-1).tolist())
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.reshape(-1)[0])
        if value.shape[0] > index:
            return json.dumps(np.asarray(value[index]).reshape(-1).tolist())
        return json.dumps(value.reshape(-1).tolist())
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def save_gray(path: Path, image_01: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_u8 = np.round(np.clip(image_01, 0.0, 1.0) * 255.0).astype(np.uint8)
    # np.rot90 is display-only; inference is performed in native NIfTI array order.
    Image.fromarray(np.rot90(image_u8), mode="L").save(path)


def save_panel(
    path: Path,
    input_hu: np.ndarray,
    vanilla_hu: np.ndarray,
    gate_hu: np.ndarray,
    metal_mask: np.ndarray,
    soft_window: Tuple[float, float],
    bone_window: Tuple[float, float],
    diff_limit: float,
    title: str,
) -> None:
    soft = [window_image(x, *soft_window) for x in (input_hu, vanilla_hu, gate_hu)]
    bone = [window_image(x, *bone_window) for x in (input_hu, vanilla_hu, gate_hu)]
    difference = np.abs(gate_hu - vanilla_hu)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    labels = ["Clinical input", "Vanilla", "MARMamba-Gate"]
    for column, (image, label) in enumerate(zip(soft, labels)):
        axes[0, column].imshow(np.rot90(image), cmap="gray", vmin=0, vmax=1)
        axes[0, column].set_title(f"{label} | soft tissue")
    im = axes[0, 3].imshow(
        np.rot90(difference), cmap="magma", vmin=0, vmax=diff_limit
    )
    axes[0, 3].set_title(f"|Gate - Vanilla| (0-{diff_limit:g} HU)")
    fig.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)

    for column, (image, label) in enumerate(zip(bone, labels)):
        axes[1, column].imshow(np.rot90(image), cmap="gray", vmin=0, vmax=1)
        axes[1, column].set_title(f"{label} | bone")
    axes[1, 3].imshow(np.rot90(metal_mask.astype(np.float32)), cmap="gray", vmin=0, vmax=1)
    axes[1, 3].set_title("Metal mask (> threshold HU)")

    for axis in axes.ravel():
        axis.axis("off")
    fig.suptitle(title, fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def batches(items: Sequence[int], size: int) -> Iterable[Sequence[int]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> None:
    args = parse_args()
    if args.top_k_per_volume <= 0:
        raise ValueError("--top_k_per_volume must be positive.")
    if args.min_slice_gap < 0:
        raise ValueError("--min_slice_gap must be non-negative.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.panel_only and args.save_npz:
        raise ValueError("Use either --panel_only or --save_npz, not both.")

    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    repo_dir = os.path.abspath(args.repo_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    from model.mamba import MambaFormer
    from model.mamba_multiscale_coarse_gate import GatedMultiScaleCoarseRefinedMambaFormer

    vanilla = MambaFormer(in_channels=1)
    gate = GatedMultiScaleCoarseRefinedMambaFormer(
        in_channels=1,
        hidden_channels=args.hidden_channels,
        gate_hidden=args.gate_hidden,
        gate_range=args.gate_range,
    )
    print("Loading Vanilla:", args.vanilla_checkpoint)
    load_model_checkpoint(vanilla, args.vanilla_checkpoint, device)
    print("Loading Gate:", args.gate_checkpoint)
    load_model_checkpoint(gate, args.gate_checkpoint, device)

    files = discover_nifti_files(args.data_path)
    if args.max_volumes is not None:
        files = files[: args.max_volumes]
    print(f"Found {len(files)} CLINIC-metal volumes")

    output_root = Path(args.save_path)
    output_root.mkdir(parents=True, exist_ok=True)
    preserve_metal = not args.no_preserve_metal
    rows: List[Dict[str, Any]] = []

    for volume_index, nifti_path in enumerate(files):
        volume_name = os.path.basename(nifti_path)
        case_id = volume_name
        if case_id.endswith(".nii.gz"):
            case_id = case_id[:-7]
        elif case_id.endswith(".nii"):
            case_id = case_id[:-4]

        nii = nib.load(nifti_path)
        volume = np.asarray(nii.dataobj, dtype=np.float32)
        if volume.ndim != 3 or volume.shape[0:2] != (512, 512):
            raise ValueError(
                f"Expected a 512x512xZ volume, got {volume.shape} for {nifti_path}"
            )

        metal_scores = np.sum(volume > args.metal_threshold, axis=(0, 1))
        selected = select_diverse_slices(
            metal_scores,
            top_k=args.top_k_per_volume,
            min_gap=args.min_slice_gap,
        )
        print(
            f"[{volume_index + 1:02d}/{len(files):02d}] {case_id}: "
            f"shape={volume.shape}, selected={selected}"
        )

        for selected_batch in batches(selected, args.batch_size):
            slices_hu = [volume[:, :, z] for z in selected_batch]
            input_tensor = tensorize_slices(slices_hu, device)

            with torch.inference_mode():
                vanilla_tensor = vanilla(input_tensor)
                try:
                    gate_raw = gate(input_tensor, return_refinement=True)
                except TypeError:
                    gate_raw = gate(input_tensor)
                gate_tensor, gate_stats = unwrap_gate_output(gate_raw)

            vanilla_units = np.clip(
                vanilla_tensor.detach().float().cpu().numpy()[:, 0], 0.0, 1.0
            )
            gate_units = np.clip(
                gate_tensor.detach().float().cpu().numpy()[:, 0], 0.0, 1.0
            )

            for batch_index, z in enumerate(selected_batch):
                input_hu = slices_hu[batch_index]
                metal_mask = input_hu > args.metal_threshold
                vanilla_hu = unit_to_hu(vanilla_units[batch_index])
                gate_hu = unit_to_hu(gate_units[batch_index])

                if preserve_metal:
                    vanilla_hu = np.where(metal_mask, input_hu, vanilla_hu)
                    gate_hu = np.where(metal_mask, input_hu, gate_hu)

                soft = tuple(float(x) for x in args.soft_window)
                bone = tuple(float(x) for x in args.bone_window)
                difference = np.abs(gate_hu - vanilla_hu)

                if not args.panel_only:
                    slice_dir = output_root / "cases" / case_id / f"slice_{z:04d}"
                    slice_dir.mkdir(parents=True, exist_ok=True)
                    for label, image_hu in (
                        ("input", input_hu),
                        ("vanilla", vanilla_hu),
                        ("gate", gate_hu),
                    ):
                        save_gray(
                            slice_dir / f"{label}_soft.png",
                            window_image(image_hu, *soft),
                        )
                        save_gray(
                            slice_dir / f"{label}_bone.png",
                            window_image(image_hu, *bone),
                        )
                    save_gray(
                        slice_dir / "abs_gate_minus_vanilla.png",
                        np.clip(difference / args.difference_limit_hu, 0.0, 1.0),
                    )
                    save_gray(
                        slice_dir / "metal_mask.png",
                        metal_mask.astype(np.float32),
                    )

                title = (
                    f"{case_id} | axial slice {z} | "
                    f"metal voxels={int(metal_scores[z])}"
                )
                save_panel(
                    output_root / "panels" / f"{case_id}_slice_{z:04d}.png",
                    input_hu,
                    vanilla_hu,
                    gate_hu,
                    metal_mask,
                    soft,
                    bone,
                    args.difference_limit_hu,
                    title,
                )

                if args.save_npz:
                    np.savez_compressed(
                        slice_dir / "arrays_hu.npz",
                        input_hu=input_hu.astype(np.float32),
                        vanilla_hu=vanilla_hu.astype(np.float32),
                        gate_hu=gate_hu.astype(np.float32),
                        metal_mask=metal_mask.astype(np.uint8),
                    )

                non_metal = ~metal_mask
                row: Dict[str, Any] = {
                    "case_id": case_id,
                    "nifti_path": nifti_path,
                    "slice_index": int(z),
                    "shape": "x".join(str(x) for x in volume.shape),
                    "spacing": json.dumps([float(x) for x in nii.header.get_zooms()[:3]]),
                    "orientation": "".join(nib.aff2axcodes(nii.affine)),
                    "metal_threshold_hu": float(args.metal_threshold),
                    "metal_voxels": int(metal_scores[z]),
                    "preserve_metal_for_visualization": preserve_metal,
                    "mean_abs_gate_minus_vanilla_hu_non_metal": float(
                        difference[non_metal].mean()
                    ),
                    "max_abs_gate_minus_vanilla_hu_non_metal": float(
                        difference[non_metal].max()
                    ),
                }
                for key, value in gate_stats.items():
                    row[f"gate_{key}"] = scalar_for_item(value, batch_index)
                rows.append(row)

        del volume

    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(output_root / "metadata.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    run_config = vars(args).copy()
    run_config.update(
        {
            "device": str(device),
            "num_volumes": len(files),
            "num_selected_slices": len(rows),
            "normalization": "clip HU [-1000,3000] -> [0,1] -> [-1,1]",
            "reference_metrics_computed": False,
        }
    )
    with open(output_root / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, ensure_ascii=False)

    print("\nFinished.")
    print("Selected slices:", len(rows))
    print("Panels:", output_root / "panels")
    print("Metadata:", output_root / "metadata.csv")


if __name__ == "__main__":
    main()
