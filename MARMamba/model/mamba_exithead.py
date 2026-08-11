"""
mamba_exithead.py
====================
File RIENG BIET cho toan bo phan cai thien ExitHead (Early-Exit / Dynamic-Depth
Inference) -- KHONG dung cham gi toi model/mamba.py goc cua paper.

Dung ky thuat KE THUA (subclass): MambaFormerWithExit ke thua nguyen MambaFormer
goc, chi THEM method moi forward_get_intermediate(), khong sua/ghi de bat ky
method nao san co. forward() goc van hoat dong CHINH XAC nhu ban dau.

Gom 3 phan:
  1. MambaFormerWithExit  -- subclass, them kha nang lay feature trung gian
  2. ExitHead              -- module du doan anh tam + do tin cay tai moi diem thoat
  3. compute_confidence_label -- tu tao nhan confidence (khong can gan tay)

CACH DUNG (trong train_exit_heads.py hoac script khac):
    from mamba_exithead import MambaFormerWithExit, ExitHead, compute_confidence_label

    net = MambaFormerWithExit(in_channels=1)
    net.load_state_dict(torch.load(checkpoint_path))   # load DUNG checkpoint cu,
                                                          # vi kien truc backbone
                                                          # khong doi gi ca
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mamba import MambaFormer


# ============================================================
# 1. SUBCLASS -- them kha nang lay feature trung gian
# ============================================================
class MambaFormerWithExit(MambaFormer):
    """
    Ke thua MambaFormer goc. forward() GIU NGUYEN khong doi (dung cho inference
    binh thuong / cac hoat dong khac nhu cu). Chi THEM forward_get_intermediate()
    de phuc vu rieng cho viec train ExitHead.

    Vi ke thua, checkpoint da train (state_dict) cua MambaFormer goc LOAD DUOC
    TRUC TIEP vao class nay ma khong can chuyen doi gi (kien truc cac layer
    khong doi, chi them method moi khong co tham so nao).
    """

    def forward_get_intermediate(self, x):
        """
        Giong het logic ben trong forward() goc, nhung dung lai o 3 diem
        (sau down1_stage / down2_stage / down3_stage) de tra ve feature
        trung gian, thay vi di tiep het toan bo encoder-decoder.

        Tra ve:
            intermediates: list 3 tuple (feature_tensor, downsample_factor)
                            downsample_factor: 1, 2, 4
            x_ori:  anh input GOC (truoc padding) -- can de ExitHead cong residual
            ori_h, ori_w: kich thuoc goc -- can de crop dung sau khi ExitHead decode
        """
        x_ori = x.clone()
        _, _, ori_h, ori_w = x.shape
        x = self.pad_to_multiple_of_eight(x)
        x = self.embedder(x)

        intermediates = []

        # Diem thoat 1: sau down1_stage, do phan giai day du
        x = self.down1_stage(x)
        intermediates.append((x.clone(), 1))
        x = self.down1(x)

        # Diem thoat 2: sau down2_stage, do phan giai 1/2
        x = self.down2_stage(x)
        intermediates.append((x.clone(), 2))
        x = self.down2(x)

        # Diem thoat 3: sau down3_stage, do phan giai 1/4
        x = self.down3_stage(x)
        intermediates.append((x.clone(), 4))
        # Khong di tiep den bottleneck (down4_stage) -- thoat o do khong
        # tiet kiem duoc nhieu tinh toan dang ke, khong dang lam diem thoat rieng.

        return intermediates, x_ori, ori_h, ori_w


# ============================================================
# 2. EXIT HEAD -- du doan anh tam + do tin cay tai 1 diem thoat
# ============================================================
class ExitHead(nn.Module):
    """
    Gan tai 1 diem thoat cu the. Tu giai ma feature ve dung do phan giai anh
    goc (upsample neu can), du doan anh phuc hoi TAM THOI + do tin cay co nen
    thoat o day khong.

    QUAN TRONG -- tuan theo dung 2 dac diem cua model goc (MambaFormer.forward()):
      1. Residual connection: cong lai x_ori truoc khi tra ve
      2. Crop dung kich thuoc goc (ori_h, ori_w) truoc khi cong residual
    Neu bo qua 1 trong 2 diem nay, early_output se SAI LECH so voi cach model
    that hoat dong, lam hong ca qua trinh tao nhan confidence.
    """

    def __init__(self, in_channels, downsample_factor, out_channels=1):
        super().__init__()
        self.downsample_factor = downsample_factor

        if downsample_factor == 1:
            self.decode = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        else:
            layers = []
            cur_channels = in_channels
            n_upsamples = int(torch.log2(torch.tensor(float(downsample_factor))).item())
            for _ in range(n_upsamples):
                next_channels = max(cur_channels // 2, out_channels * 4)
                layers.append(nn.Conv2d(cur_channels, next_channels, kernel_size=3, padding=1))
                layers.append(nn.GELU())
                layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
                cur_channels = next_channels
            layers.append(nn.Conv2d(cur_channels, out_channels, kernel_size=3, padding=1))
            self.decode = nn.Sequential(*layers)

        self.confidence = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(in_channels // 4, 4), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(max(in_channels // 4, 4), 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, feat, x_ori, ori_h, ori_w):
        """
        feat: feature tai diem thoat nay, shape [B, C, h, w]
        x_ori: anh input GOC (truoc padding), shape [B, 1, ori_h, ori_w]

        Tra ve: (early_output [B,1,ori_h,ori_w], confidence [B])
        """
        decoded = self.decode(feat)
        decoded = decoded[:, :, :ori_h, :ori_w]
        early_output = decoded + x_ori

        confidence = self.confidence(feat).view(-1)
        return early_output, confidence


# ============================================================
# 3. TU TAO NHAN CONFIDENCE -- khong can gan tay
# ============================================================
def compute_confidence_label(early_output, full_output, gt, psnr_gap_threshold=0.5):
    """
    So sanh PSNR cua early_output va full_output (ca 2 so voi gt).
    Neu gap nho (early gan bang full) -> nhan=1 (nen thoat som o day).
    Nguoc lai -> nhan=0 (chua du tot, can di sau hon).

    early_output, full_output, gt: tensor [B, 1, H, W]
    Tra ve: tensor [B] gia tri 0.0/1.0
    """
    def batch_psnr(pred, target, max_val=1.0, eps=1e-8):
        mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
        mse = torch.clamp(mse, min=eps)
        return 10 * torch.log10((max_val ** 2) / mse)

    psnr_early = batch_psnr(early_output, gt)
    psnr_full = batch_psnr(full_output, gt)
    gap = psnr_full - psnr_early
    return (gap < psnr_gap_threshold).float()


# ============================================================
# Self-test nhanh khi chay truc tiep file nay
# ============================================================
if __name__ == "__main__":
    print("Chay self-test cho mamba_exithead.py...")
    net = MambaFormerWithExit(in_channels=1, dim=12, depth=[1, 2, 2, 4, 1])
    x = torch.randn(2, 1, 256, 256)
    gt = torch.rand(2, 1, 256, 256)

    out_original = net(x)
    assert out_original.shape == x.shape, "forward() goc bi sai shape!"
    print(f"  forward() goc: OK, output shape={list(out_original.shape)}")

    intermediates, x_ori, ori_h, ori_w = net.forward_get_intermediate(x)
    exit_heads = nn.ModuleList([
        ExitHead(in_channels=feat.shape[1], downsample_factor=factor)
        for feat, factor in intermediates
    ])

    for i, ((feat, factor), head) in enumerate(zip(intermediates, exit_heads)):
        early_output, confidence = head(feat, x_ori, ori_h, ori_w)
        assert early_output.shape == x.shape, f"ExitHead {i+1} sai shape!"
        label = compute_confidence_label(early_output, out_original, gt)
        print(f"  ExitHead {i+1} (factor={factor}): OK, "
              f"early_output.shape={list(early_output.shape)}, "
              f"confidence={confidence.detach().tolist()}, "
              f"nhan_tu_tao={label.tolist()}")

    print("TAT CA SELF-TEST PASS")
