"""Pack FOB72 + figset1 + figset2 into one folder: 480x480 PNGs with fixed naming."""
#%%
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

fobpath = r"E:\#Stimsets\FOB72"
figset1a = r"G:\我的云端硬盘\#BLab_Works\_260427_DQInva\objShading"
figset1b = r"G:\我的云端硬盘\#BLab_Works\_260427_DQInva\objTexture"

figset2 = r"E:\#Stimsets\Depth_Cue_Invariant\faces_fruits_bodies_depth_cues"

savepath = r"E:\#Stimsets\DQInva_v260508"

SIZE = 480
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
    return out


def center_crop_square(im: Image.Image) -> Image.Image:
    w, h = im.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return im.crop((left, top, left + side, top + side))


def to_square_480(im: Image.Image) -> Image.Image:
    """Resize to 480x480 (center-crop to square first if needed)."""
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA") if "A" in im.getbands() else im.convert("RGB")
    sq = center_crop_square(im)
    return sq.resize((SIZE, SIZE), Image.Resampling.LANCZOS)


def resize_fill_480(im: Image.Image) -> Image.Image:
    """Resize to 480x480 (square crop + scale)."""
    return to_square_480(im)


def save_png(dst: Path, im: Image.Image) -> None:
    if im.mode == "RGBA":
        im.save(dst, "PNG")
    else:
        rgb = im.convert("RGB")
        rgb.save(dst, "PNG")


def main() -> None:
    out = Path(savepath)
    out.mkdir(parents=True, exist_ok=True)

    fob = Path(fobpath)
    fs1a = Path(figset1a)
    fs1b = Path(figset1b)
    fs2 = Path(figset2)

    # --- 1. FOB72: 00001.png … ---
    fob_files = list_images(fob)
    if len(fob_files) != 72:
        print(
            f"Warning: expected 72 images in {fob}, found {len(fob_files)}.",
            file=sys.stderr,
        )
    for i, src in enumerate(fob_files[:72], start=1):
        im = Image.open(src)
        im = resize_fill_480(im)
        dst = out / f"{i:05d}.png"
        save_png(dst, im)
        print(f"FOB [{i}/72] {src.name} -> {dst.name}")

    # --- 2. figset1: center-crop square 480, 10001+, two full passes ---
    fig1_list = list_images(fs1a) + list_images(fs1b)
    counter = 10001
    for _pass in range(2):
        for src in fig1_list:
            im = Image.open(src)
            im = to_square_480(im)
            dst = out / f"{counter}.png"
            save_png(dst, im)
            print(f"figset1 pass{_pass + 1} [{counter}] {src.name} -> {dst.name}")
            counter += 1

    # --- 3. figset2: 480x480, 20001+, two passes ---
    fig2_list = list_images(fs2)
    counter = 20001
    for _pass in range(2):
        for src in fig2_list:
            im = Image.open(src)
            im = resize_fill_480(im)
            dst = out / f"{counter}.png"
            save_png(dst, im)
            print(f"figset2 pass{_pass + 1} [{counter}] {src.name} -> {dst.name}")
            counter += 1

    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
