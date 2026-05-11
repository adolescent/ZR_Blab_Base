"""
Pack and rename outputs into a single _Overall folder.

Fixed mask layout (current project rule):
  E:/#Stimsets/Metamer_Demolish_v260508/Metamer_Part_Modules/<id>/mask_01.png ... mask_05.png

RGB outputs (2xxxx / 3xxxx / 4xxxx / 6xxxx):
  Load the matching image from Object_No_BK/<id>.*, apply the union of selected part masks,
  keep pixels where mask==1, set all other pixels to gray 127.
"""
#%%
from __future__ import annotations

import itertools
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

OUT_DIR = Path(r"E:\#Stimsets\Metamer_Demolish_v260508\_Overall")
FOB_DIR = Path(r"E:\#Stimsets\Metamer_NSD_FOB_v260420")
MASKS_ROOT = Path(r"E:\#Stimsets\Metamer_Demolish_v260508\Metamer_Part_Modules")
OBJECT_NO_BK_DIR = Path(r"E:\#Stimsets\Metamer_Demolish_v260508\Object_No_BK")

MASK_COUNT_PER_IMAGE = 5
BG_GRAY = 127

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
STEM_TO_DIR: dict[str, Path] = {}


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_mask_png(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img)
    return arr >= 128


def resolve_object_rgb_path(stem: str) -> Path:
    """Object_No_BK image basename must match Metamer_Part_Modules folder id (e.g. 0012.png)."""
    for ext in IMAGE_EXTS:
        cand = OBJECT_NO_BK_DIR / f"{stem}{ext}"
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"No image for id {stem!r} in {OBJECT_NO_BK_DIR} (tried {stem}<ext>)"
    )


def composite_keep_masked(rgb: np.ndarray, mask_bool: np.ndarray, bg: int = BG_GRAY) -> np.ndarray:
    """Keep RGB where mask is True; elsewhere fill with gray bg."""
    if mask_bool.shape[:2] != rgb.shape[:2]:
        mask_img = Image.fromarray(mask_bool.astype(np.uint8) * 255, mode="L")
        mask_img = mask_img.resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST)
        mask_bool = np.asarray(mask_img) >= 128
    a = mask_bool.astype(np.float32)
    fg = rgb.astype(np.float32)
    out = fg * a[..., None] + float(bg) * (1.0 - a[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def save_rgb_jpg(rgb: np.ndarray, out_path: Path) -> None:
    Image.fromarray(rgb, mode="RGB").save(out_path, quality=95)


def rename_fob_file(src: Path) -> str:
    stem = src.stem
    if len(stem) >= 1 and stem[0] == "5" and stem[1:].isdigit():
        stem = "9" + stem[1:]
    return stem + src.suffix.lower()


def copy_fob_images(dst_dir: Path) -> int:
    files = list_images(FOB_DIR)
    if not files:
        print(f"No images found in {FOB_DIR}", file=sys.stderr)
        return 0
    n = 0
    for src in tqdm(files, desc="Copy FOB"):
        dst_name = rename_fob_file(src)
        shutil.copy2(src, dst_dir / dst_name)
        n += 1
    return n


def _resolve_mask_file_with_prefix(folder: Path, prefix: str) -> Path:
    for ext in IMAGE_EXTS:
        cand = folder / f"{prefix}{ext}"
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"Missing {prefix} in {folder} (tried extensions {sorted(IMAGE_EXTS)})")


def get_mask_paths_for_stem(stem: str, n_masks: int) -> list[Path]:
    item_dir = STEM_TO_DIR.get(stem, MASKS_ROOT / stem)
    paths: list[Path] = []
    for i in range(1, n_masks + 1):
        try:
            found = _resolve_mask_file_with_prefix(item_dir, f"mask_{i:02d}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Missing masks for {stem}: expected {item_dir / f'mask_{i:02d}'}.{sorted(IMAGE_EXTS)}"
            )
        paths.append(found)
    return paths


def load_masks_for_stem(stem: str, n_masks: int) -> list[np.ndarray]:
    paths = get_mask_paths_for_stem(stem, n_masks)
    return [read_mask_png(p) for p in paths]


def union_masks(masks: list[np.ndarray], idxs: tuple[int, ...]) -> np.ndarray:
    out = None
    for i in idxs:
        if out is None:
            out = masks[i].copy()
        else:
            out |= masks[i]
    if out is None:
        raise ValueError("Empty mask selection")
    return out


def pack_combinations(mask_stems: list[str], k: int, prefix: int, start_index: int, n_masks: int) -> int:
    """
    For each image stem, generate all k-combinations of n_masks masks.
    Composite onto Object_No_BK/<stem>.*: keep union mask region, gray 127 elsewhere.
    """
    idx = start_index
    for stem in tqdm(mask_stems, desc=f"Pack k={k}"):
        img_path = resolve_object_rgb_path(stem)
        rgb = np.asarray(Image.open(img_path).convert("RGB"))
        masks = load_masks_for_stem(stem, n_masks)
        for comb in itertools.combinations(range(n_masks), k):
            out_mask = union_masks(masks, comb)
            out_rgb = composite_keep_masked(rgb, out_mask)
            out_name = f"{prefix}{idx:04d}.jpg"
            save_rgb_jpg(out_rgb, OUT_DIR / out_name)
            idx += 1
    return idx - start_index


def copy_object_no_bk(dst_dir: Path, prefix: int, start_index: int) -> int:
    files = list_images(OBJECT_NO_BK_DIR)
    if not files:
        print(f"No images found in {OBJECT_NO_BK_DIR}", file=sys.stderr)
        return 0
    idx = start_index
    n = 0
    for src in tqdm(files, desc="Copy Object_No_BK"):
        dst = dst_dir / f"{prefix}{idx:04d}.jpg"
        shutil.copy2(src, dst)
        idx += 1
        n += 1
    return n


def detect_mask_layout_and_stems() -> tuple[list[str], int]:
    """Return sorted image ids under fixed layout: <id>/mask_01...mask_05."""
    global STEM_TO_DIR
    STEM_TO_DIR = {}

    if not MASKS_ROOT.is_dir():
        raise FileNotFoundError(f"MASKS_ROOT not found: {MASKS_ROOT}")
    stems: list[str] = []
    for p in sorted([x for x in MASKS_ROOT.iterdir() if x.is_dir()]):
        ok = True
        for i in range(1, MASK_COUNT_PER_IMAGE + 1):
            try:
                _resolve_mask_file_with_prefix(p, f"mask_{i:02d}")
            except FileNotFoundError:
                ok = False
                break
        if ok:
            STEM_TO_DIR[p.name] = p
            stems.append(p.name)

    if not stems:
        raise FileNotFoundError(
            f"No valid id folders found under {MASKS_ROOT}. "
            f"Expected folders like <id>/mask_01.png ... mask_05.png"
        )
    print(f"Mask layout fixed: found {len(stems)} id folders under {MASKS_ROOT}.")
    return stems, MASK_COUNT_PER_IMAGE


def main() -> None:
    ensure_dir(OUT_DIR)

    n_fob = copy_fob_images(OUT_DIR)

    mask_stems, n_masks = detect_mask_layout_and_stems()
    print(f"Found {len(mask_stems)} image stems; {n_masks} masks per image.")

    n_1 = pack_combinations(mask_stems, k=1, prefix=2, start_index=1, n_masks=n_masks)
    n_2 = pack_combinations(mask_stems, k=2, prefix=3, start_index=1, n_masks=n_masks)
    n_3 = pack_combinations(mask_stems, k=3, prefix=4, start_index=1, n_masks=n_masks)
    n_4 = pack_combinations(mask_stems, k=4, prefix=6, start_index=1, n_masks=n_masks)

    n_obj = copy_object_no_bk(OUT_DIR, prefix=7, start_index=1)

    print(
        "Done.\n"
        f"- FOB copied: {n_fob}\n"
        f"- Single-part composites Object_No_BK+mask (2xxxx): {n_1}\n"
        f"- 2-part union composites (3xxxx): {n_2}\n"
        f"- 3-part union composites (4xxxx): {n_3}\n"
        f"- 4-part union composites (6xxxx): {n_4}\n"
        f"- Object_No_BK copied (7xxxx): {n_obj}\n"
        f"Output: {OUT_DIR}"
    )


if __name__ == "__main__":
    main()

