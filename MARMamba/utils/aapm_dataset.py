"""
aapm_dataset.py
================
Drop-in replacement for MARMamba's utils/dataset.py MARTrainDataset — đọc
trực tiếp từ AAPM CT-MAR raw export, hỗ trợ CẤU TRÚC NHIỀU VÙNG GIẢI PHẪU:

    train_data_dir/              <- trỏ -train_data_dir vào ĐÚNG folder cha này
    ├── head1/
    │   ├── Baseline/  training_head_metalart_img<ID>_512x512x1.raw
    │   ├── Target/    training_head_nometal_img<ID>_512x512x1.raw
    │   └── Mask/
    ├── head2/
    │   ├── Baseline/
    │   ├── Target/
    │   └── Mask/
    ├── chest1/
    │   ├── Baseline/
    │   ├── Target/
    │   └── Mask/
    ├── abdomen1/
    │   └── ...
    └── ...

Code tự động quét TẤT CẢ folder con có chứa Baseline/ + Target/ bên trong
train_data_dir, gộp lại thành 1 dataset lớn duy nhất — không cần sửa
train_step.py để nhận list nhiều path, chỉ cần trỏ đúng 1 path folder cha.

Vì mỗi vùng giải phẫu khác nhau (head1, head2, chest1...) có ID ảnh có thể
trùng nhau (ví dụ cả head1 và chest1 đều có img1001), code prefix ID theo
tên folder con để tránh nhầm lẫn.
"""

import os
import re
import glob
import random
import time
import numpy as np
import torch
import torch.utils.data as udata
import torchvision.utils as tvu
from PIL import Image
from numpy.random import RandomState
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize

HU_MIN, HU_MAX = -1000.0, 3000.0
MASK_HU_THRESH = 2500.0

FNAME_ID_RE = re.compile(r"img(\d+)")
DIMS_RE = re.compile(r"(\d+)x(\d+)x(\d+)")


def parse_dims(filename: str):
    m = DIMS_RE.search(filename)
    if not m:
        raise ValueError(f"Không đọc được kích thước từ tên file: {filename}")
    w, h, d = (int(x) for x in m.groups())
    return h, w, d


def load_raw(path: str, dtype=np.float32) -> np.ndarray:
    rows, cols, slices = parse_dims(os.path.basename(path))
    arr = np.fromfile(path, dtype=dtype)
    expected = rows * cols * slices
    if arr.size != expected:
        raise ValueError(f"{path}: có {arr.size} phần tử, cần {expected}")
    return arr.reshape(rows, cols) if slices == 1 else arr.reshape(slices, rows, cols)


def hu_to_unit(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, HU_MIN, HU_MAX)
    return ((img - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def _find_anatomy_folders(parent_dir):
    """Tìm tất cả folder con chứa cả Baseline/ và Target/ bên trong."""
    found = []
    if os.path.isdir(os.path.join(parent_dir, "Baseline")) and os.path.isdir(os.path.join(parent_dir, "Target")):
        found.append(parent_dir)
        return found
    for entry in sorted(os.listdir(parent_dir)):
        sub = os.path.join(parent_dir, entry)
        if not os.path.isdir(sub):
            continue
        if os.path.isdir(os.path.join(sub, "Baseline")) and os.path.isdir(os.path.join(sub, "Target")):
            found.append(sub)
    return found


def _index_one_anatomy(anatomy_dir):
    """Index 1 folder giải phẫu (vd head1/), trả về list sample dict."""
    name_prefix = os.path.basename(os.path.normpath(anatomy_dir))
    baseline_dir = os.path.join(anatomy_dir, "Baseline")
    target_dir = os.path.join(anatomy_dir, "Target")
    mask_dir = os.path.join(anatomy_dir, "Mask")

    baseline_files = [f for f in glob.glob(os.path.join(baseline_dir, "*.raw")) if "img" in os.path.basename(f)]
    target_files = {
        FNAME_ID_RE.search(os.path.basename(f)).group(1): f
        for f in glob.glob(os.path.join(target_dir, "*.raw"))
        if "img" in os.path.basename(f) and FNAME_ID_RE.search(os.path.basename(f))
    }
    mask_files = {}
    if os.path.isdir(mask_dir):
        for f in glob.glob(os.path.join(mask_dir, "*")):
            m = FNAME_ID_RE.search(os.path.basename(f))
            if m:
                mask_files[m.group(1)] = f

    samples = []
    for bf in baseline_files:
        m = FNAME_ID_RE.search(os.path.basename(bf))
        if not m:
            continue
        img_id = m.group(1)
        if img_id not in target_files:
            continue
        samples.append({
            "id": f"{name_prefix}_{img_id}",
            "baseline": bf,
            "target": target_files[img_id],
            "mask": mask_files.get(img_id),
        })
    return samples


class AAPMTrainDataset(udata.Dataset):
    """
    train_data_dir: 1 trong 2 dạng
      - folder giải phẫu đơn (có Baseline/Target trực tiếp bên trong)
      - folder cha chứa nhiều folder giải phẫu con (head1, head2, chest1...)
    Code tự phát hiện dạng nào và gộp toàn bộ sample lại thành 1 dataset.
    """

    def __init__(self, crop_size, train_data_dir, random_flip, random_rotate):
        super().__init__()
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.crop_size = crop_size

        anatomy_dirs = _find_anatomy_folders(train_data_dir)
        if not anatomy_dirs:
            raise FileNotFoundError(
                f"Không tìm thấy folder Baseline/Target nào trong hoặc bên trong: {train_data_dir}"
            )

        self.samples = []
        print(f"Tìm thấy {len(anatomy_dirs)} vùng giải phẫu:")
        for d in anatomy_dirs:
            found = _index_one_anatomy(d)
            print(f"  {os.path.basename(d)}: {len(found)} cặp")
            self.samples.extend(found)

        self.file_num = len(self.samples)
        if self.file_num == 0:
            raise RuntimeError(f"Không tìm được cặp baseline/Target nào trong: {train_data_dir}")
        print(f"Tổng cộng: {self.file_num} cặp từ {len(anatomy_dirs)} vùng giải phẫu")
        self.rand_state = RandomState(66)

    def __len__(self):
        return self.file_num

    def _resize_if_needed(self, input_img, gt_img, crop_width, crop_height):
        h, w = input_img.shape
        if h >= crop_height and w >= crop_width:
            return input_img, gt_img
        new_h, new_w = max(h, crop_height), max(w, crop_width)
        input_img = np.array(Image.fromarray(input_img).resize((new_w, new_h), Image.LANCZOS))
        gt_img = np.array(Image.fromarray(gt_img).resize((new_w, new_h), Image.LANCZOS))
        return input_img, gt_img

    def __getitem__(self, idx):
        crop_width, crop_height = self.crop_size
        sample = self.samples[idx]

        XLI = hu_to_unit(load_raw(sample["baseline"]))
        Xgt = hu_to_unit(load_raw(sample["target"]))

        input_img, gt_img = self._resize_if_needed(XLI, Xgt, crop_width, crop_height)

        height, width = input_img.shape
        x = randrange(0, width - crop_width + 1)
        y = randrange(0, height - crop_height + 1)
        input_crop_img = input_img[y:y + crop_height, x:x + crop_width]
        gt_crop_img = gt_img[y:y + crop_height, x:x + crop_width]

        transform_input = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
        transform_gt = Compose([ToTensor()])
        input_im = transform_input(input_crop_img)
        gt = transform_gt(gt_crop_img)

        if self.random_flip and random.random() < 0.5:
            input_im = torch.flip(input_im, dims=[-1])
            gt = torch.flip(gt, dims=[-1])

        if self.random_rotate:
            r = random.random()
            if r < 0.25:
                input_im = torch.rot90(input_im, k=1, dims=(1, 2)); gt = torch.rot90(gt, k=1, dims=(1, 2))
            elif r < 0.5:
                input_im = torch.rot90(input_im, k=3, dims=(1, 2)); gt = torch.rot90(gt, k=3, dims=(1, 2))
            elif r < 0.75:
                input_im = torch.rot90(input_im, k=2, dims=(1, 2)); gt = torch.rot90(gt, k=2, dims=(1, 2))

        return input_im, gt


def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    tvu.save_image(img, file_directory)


def test_image(data_path, model):
    anatomy_dirs = _find_anatomy_folders(data_path)
    transform_input = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
    transform_gt = Compose([ToTensor()])

    total = 0
    total_time = 0.0
    for d in anatomy_dirs:
        samples = _index_one_anatomy(d)
        for s in samples:
            XLI = hu_to_unit(load_raw(s["baseline"]))
            Xgt = hu_to_unit(load_raw(s["target"]))

            input_t = transform_input(XLI).unsqueeze(0).cuda()
            gt_t = transform_gt(Xgt).cuda()

            start_time = time.time()
            output = model(input_t)
            total_time += time.time() - start_time
            total += 1

            save_image(gt_t, os.path.join("./eva/gt", f"{s['id']}.png"))
            save_image(output, os.path.join("./eva/output", f"{s['id']}.png"))

    return total, (total_time / total if total else 0.0)


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    ds = AAPMTrainDataset(crop_size=(256, 256), train_data_dir=root,
                          random_flip=True, random_rotate=True)
    print(f"len(ds) = {len(ds)}")
    x, y = ds[0]
    print(f"input shape={x.shape}, gt shape={y.shape}")