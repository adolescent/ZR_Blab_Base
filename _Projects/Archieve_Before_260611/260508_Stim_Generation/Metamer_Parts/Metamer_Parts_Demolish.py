"""
Remove background with Hugging Face briaai/RMBG-1.4 and composite onto gray 127.

Requires PyTorch. RMBG's custom BriaRMBG weights load correctly with transformers 4.x;
if `pipeline(...)` fails with missing `all_tied_weights_keys`, run:
  pip install "transformers>=4.44,<5"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import pipeline

raw_parts = r"E:\#Stimsets\Raw_Objects"

save_path = r"E:\#Stimsets\Metamer_Part_Demolish"

BG_GRAY = 127

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTS)


def _composite_rgb_gray(rgb: np.ndarray, alpha_hw: np.ndarray) -> np.ndarray:
    """rgb, alpha uint8 or float; alpha HxW in [0,1] or 0-255."""
    if alpha_hw.dtype != np.float32:
        a = alpha_hw.astype(np.float32)
        if a.max() > 1.0:
            a = a / 255.0
    else:
        a = alpha_hw
        if a.max() > 1.0:
            a = a / 255.0
    a = np.clip(a, 0.0, 1.0)
    fg = rgb.astype(np.float32)
    out = fg * a[..., None] + float(BG_GRAY) * (1.0 - a[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def process_folder(
    src_dir: str | Path,
    dst_dir: str | Path,
    pipe,
) -> None:
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    files = _list_images(src)
    if not files:
        print(f"No images found in {src}", file=sys.stderr)
        return

    for path in tqdm(files, desc="RMBG"):
        path = path.resolve()
        try:
            img = Image.open(path).convert("RGB")
            mask = pipe(str(path), return_mask=True)
            if not isinstance(mask, Image.Image):
                raise TypeError(f"expected PIL mask, got {type(mask)}")
            if mask.size != img.size:
                mask = mask.resize(img.size, Image.Resampling.BILINEAR)
            if mask.mode != "L":
                mask = mask.convert("L")
            rgb = np.asarray(img)
            alpha = np.asarray(mask, dtype=np.float32) / 255.0
            out = _composite_rgb_gray(rgb, alpha)
            out_path = dst / f"{path.stem}.png"
            Image.fromarray(out).save(out_path)
        except Exception as e:
            print(f"skip {path}: {e}", file=sys.stderr)


def main() -> None:
    dev = 0 if torch.cuda.is_available() else -1
    try:
        pipe = pipeline(
            "image-segmentation",
            model="briaai/RMBG-1.4",
            trust_remote_code=True,
            device=dev,
        )
    except AttributeError as e:
        print(
            "Failed to load RMBG-1.4 (often transformers 5.x vs custom model). "
            'Try: pip install "transformers>=4.44,<5"\n'
            f"Original error: {e}",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    process_folder(raw_parts, save_path, pipe)


if __name__ == "__main__":
    main()
