"""
AAPM CT-MAR dataset for Exp1: Metal-Geometry Guided Dynamic FMB.

This file intentionally does NOT modify utils/aapm_dataset.py.
It reuses the baseline indexing/loading helpers and only adds metal-mask loading,
synchronized augmentation, and an Exp1 evaluation helper.
"""

import os
import random
import time

import numpy as np
import torch
import torchvision.utils as tvu
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize

from utils.aapm_dataset import (
    AAPMTrainDataset,
    _find_anatomy_folders,
    _index_one_anatomy,
    hu_to_unit,
    load_raw,
)


class AAPMTrainDatasetExp1(AAPMTrainDataset):
    """AAPM loader that returns (artifact_image, clean_gt, metal_mask).

    The crop/flip/rotation applied to the mask is exactly the same as the one
    applied to the artifact image and target image.
    """

    def _resize_triplet_if_needed(self, input_img, gt_img, mask_img, crop_width, crop_height):
        h, w = input_img.shape
        if h >= crop_height and w >= crop_width:
            return input_img, gt_img, mask_img

        new_h, new_w = max(h, crop_height), max(w, crop_width)
        input_img = np.array(
            Image.fromarray(input_img).resize((new_w, new_h), Image.LANCZOS),
            dtype=np.float32,
        )
        gt_img = np.array(
            Image.fromarray(gt_img).resize((new_w, new_h), Image.LANCZOS),
            dtype=np.float32,
        )
        # Binary masks must never be interpolated with bilinear/Lanczos.
        mask_img = np.array(
            Image.fromarray(mask_img).resize((new_w, new_h), Image.NEAREST),
            dtype=np.float32,
        )
        mask_img = (mask_img > 0.5).astype(np.float32)
        return input_img, gt_img, mask_img

    def __getitem__(self, idx):
        crop_width, crop_height = self.crop_size
        sample = self.samples[idx]

        if sample.get("mask") is None:
            raise FileNotFoundError(
                f"Exp1 requires a metal mask, but no Mask file was indexed for sample {sample['id']}"
            )

        input_img = hu_to_unit(load_raw(sample["baseline"]))
        gt_img = hu_to_unit(load_raw(sample["target"]))

        # AAPM metal-only image mask is stored as float raw data.  Binarize it
        # explicitly so the guidance network always receives {0, 1} values.
        mask_img = load_raw(sample["mask"]).astype(np.float32)
        if mask_img.ndim == 3:
            mask_img = np.squeeze(mask_img)
        mask_img = (mask_img != 0).astype(np.float32)

        input_img, gt_img, mask_img = self._resize_triplet_if_needed(
            input_img, gt_img, mask_img, crop_width, crop_height
        )

        height, width = input_img.shape
        x = randrange(0, width - crop_width + 1)
        y = randrange(0, height - crop_height + 1)

        input_crop = input_img[y:y + crop_height, x:x + crop_width]
        gt_crop = gt_img[y:y + crop_height, x:x + crop_width]
        mask_crop = mask_img[y:y + crop_height, x:x + crop_width]

        transform_input = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
        transform_gt = Compose([ToTensor()])
        transform_mask = Compose([ToTensor()])

        input_t = transform_input(input_crop)
        gt_t = transform_gt(gt_crop)
        mask_t = transform_mask(mask_crop).float()

        # IMPORTANT: all spatial augmentation decisions are shared by image,
        # target, and mask.  Otherwise the geometry guidance becomes invalid.
        if self.random_flip and random.random() < 0.5:
            input_t = torch.flip(input_t, dims=[-1])
            gt_t = torch.flip(gt_t, dims=[-1])
            mask_t = torch.flip(mask_t, dims=[-1])

        if self.random_rotate:
            r = random.random()
            if r < 0.25:
                k = 1
            elif r < 0.5:
                k = 3
            elif r < 0.75:
                k = 2
            else:
                k = 0

            if k:
                input_t = torch.rot90(input_t, k=k, dims=(1, 2))
                gt_t = torch.rot90(gt_t, k=k, dims=(1, 2))
                mask_t = torch.rot90(mask_t, k=k, dims=(1, 2))

        return input_t, gt_t, mask_t


def save_image(img, file_directory):
    os.makedirs(os.path.dirname(file_directory), exist_ok=True)
    tvu.save_image(img, file_directory)


def test_image_exp1(data_path, model, save_root="./eva_exp1"):
    """Evaluate Exp1 using the AAPM metal mask as the second model input."""
    anatomy_dirs = _find_anatomy_folders(data_path)
    transform_input = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
    transform_gt = Compose([ToTensor()])
    transform_mask = Compose([ToTensor()])

    gt_dir = os.path.join(save_root, "gt")
    output_dir = os.path.join(save_root, "output")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    total = 0
    total_time = 0.0

    for anatomy_dir in anatomy_dirs:
        samples = _index_one_anatomy(anatomy_dir)
        for sample in samples:
            if sample.get("mask") is None:
                raise FileNotFoundError(
                    f"Exp1 requires a metal mask, but no Mask file was indexed for sample {sample['id']}"
                )

            input_img = hu_to_unit(load_raw(sample["baseline"]))
            gt_img = hu_to_unit(load_raw(sample["target"]))
            mask_img = load_raw(sample["mask"]).astype(np.float32)
            if mask_img.ndim == 3:
                mask_img = np.squeeze(mask_img)
            mask_img = (mask_img != 0).astype(np.float32)

            input_t = transform_input(input_img).unsqueeze(0).cuda()
            gt_t = transform_gt(gt_img).cuda()
            mask_t = transform_mask(mask_img).unsqueeze(0).float().cuda()

            start_time = time.time()
            output = model(input_t, mask_t)
            total_time += time.time() - start_time
            total += 1

            save_image(gt_t, os.path.join(gt_dir, f"{sample['id']}.png"))
            save_image(output, os.path.join(output_dir, f"{sample['id']}.png"))

    return total, (total_time / total if total else 0.0)


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "."
    ds = AAPMTrainDatasetExp1(
        crop_size=(256, 256),
        train_data_dir=root,
        random_flip=True,
        random_rotate=True,
    )
    print(f"len(ds) = {len(ds)}")
    x, y, m = ds[0]
    print(f"input={x.shape}, gt={y.shape}, mask={m.shape}, mask values={torch.unique(m)}")
