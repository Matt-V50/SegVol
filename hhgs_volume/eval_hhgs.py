"""
SegVol (point prompt) on HHGS cases with the HHGS clicks (2026-11-13). Same procedure as the r2 baseline script
(eval_seg_3D_segvol.py of the r2 rebuttal, functions below taken from it):
  - SegVol_v1.pth, its fixed input (32, 256, 256) with patch (4, 16, 16); grid axes (X, Y, Z) -> model (D, H, W) =
    (Z, X, Y)
  - per click a window of 128 x 128 (X, Y) x 16 (Z) voxels centred on the click (zero-padded outside the volume;
    16 = 128 / 8: the same sampling rate on the three axes after resizing to (32, 256, 256)), SegVol's own
    ForegroundNormalization + MinMaxNormalization inside the window, trilinear resize
  - one positive point at the window centre, sigmoid probability resized back, >= 0.5
The image is the HHGS volume vol_<vol>.npy (rc_v). Scored by evaluating/baselines/common.py of the hhgs repo (Dice
with the clicked GT node over the valid voxels).

usage:  python hhgs_volume/eval_hhgs.py --case <DATA_ROOT case> --out <OUT_ROOT case>
        (environment segvol: monai 1.2.0, transformers 4.56.0; run from the SegVol repo root)
output: <out>/baselines/segvol.json
"""
import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
os.chdir(HERE.parent)                                    # ./config/clip
from segment_anything_volumetric import sam_model_registry  # noqa: E402
from network.model import SegVol  # noqa: E402

spec = importlib.util.spec_from_file_location("hhgs_common", HERE.parents[2] / "evaluating/baselines/common.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)

MODEL_INPUT, PATCH = (32, 256, 256), (4, 16, 16)
VOL_TO_MODEL, MODEL_TO_VOL = (2, 0, 1), (1, 2, 0)
EXTENT = (128, 128, 16)                                  # window (X, Y, Z) in voxels
CKPT = "/home/matt/colab_data/content/Models/SegVol/SegVol_v1.pth"
STALE_CLIP_KEY = "text_encoder.clip_text_model.text_model.embeddings.position_ids"


def load_segvol():
    args = SimpleNamespace(spatial_size=MODEL_INPUT, patch_size=PATCH, test_mode=True)
    sam = sam_model_registry["vit"](args=args)
    model = SegVol(image_encoder=sam.image_encoder, mask_decoder=sam.mask_decoder, prompt_encoder=sam.prompt_encoder,
                   clip_ckpt="./config/clip", roi_size=MODEL_INPUT, patch_size=PATCH, test_mode=True).cuda()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in ck["model"].items()}
    sd.pop(STALE_CLIP_KEY, None)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return model.eval()


def window(center, extent, shape):
    src, dst = [], []
    for c, e, n in zip(center, extent, shape):
        lo = int(c) - e // 2
        s_lo, s_hi = max(lo, 0), min(lo + e, int(n))
        src.append(slice(s_lo, s_hi))
        dst.append(slice(s_lo - lo, s_hi - lo))
    return tuple(src), tuple(dst)


def quantile(t, q):
    if t.numel() > 2 ** 24:
        t = t[::(t.numel() + 2 ** 24 - 1) // 2 ** 24]
    return torch.quantile(t, q)


def segvol_normalize(roi):
    flat = roi.reshape(-1)
    fg = flat[flat > flat.mean()]
    if fg.numel() == 0:
        return torch.zeros_like(roi)
    lo, hi, mean, std = quantile(fg, 0.0005), quantile(fg, 0.9995), fg.mean(), fg.std()
    if not torch.isfinite(std) or float(std) < 1e-8:
        std = torch.ones_like(std)
    out = (roi.clamp(min=lo, max=hi) - mean) / std
    out = out - out.min()
    return out / out.max().clamp(min=1e-8)


@torch.no_grad()
def predict(model, vol, center):
    src, dst = window(center, EXTENT, vol.shape)
    roi = torch.zeros(EXTENT, device="cuda")
    roi[dst] = vol[src]
    roi = segvol_normalize(roi).permute(*VOL_TO_MODEL)[None, None]
    roi = F.interpolate(roi, size=MODEL_INPUT, mode="trilinear", align_corners=False)
    pts = (torch.tensor([[[m / 2.0 for m in MODEL_INPUT]]], device="cuda"), torch.ones((1, 1), device="cuda"))
    prob = torch.sigmoid(model(roi, points=pts))
    prob = F.interpolate(prob, size=tuple(EXTENT[a] for a in VOL_TO_MODEL), mode="trilinear", align_corners=False)
    m = (prob[0, 0] >= 0.5).permute(*MODEL_TO_VOL)
    return m, src, dst


def main(model, case, out):
    clicks = common.Clicks(out)
    vol = torch.as_tensor(np.load(Path(case) / f"vol_{clicks.vol}.npy").astype(np.float32), device="cuda")
    dice, t0 = [], time.time()
    for i in range(len(clicks)):
        m, src, dst = predict(model, vol, clicks.xyz[i])
        full = torch.zeros(clicks.shape, dtype=torch.bool, device="cuda")
        full[src] = m[dst]
        dice.append(clicks.dice(i, full))
    res = clicks.write_result("segvol", dice, seconds=round(time.time() - t0, 1), extent=EXTENT)
    print(f"{case}: leaf {res['leaf']}  own {res['own']:.4f}  (n {len(dice)})  [{res['seconds']} s]", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", nargs="+", required=True)
    ap.add_argument("--out", nargs="+", required=True)
    a = ap.parse_args()
    model = load_segvol()
    for c, o in zip(a.case, a.out):
        main(model, c, o)
