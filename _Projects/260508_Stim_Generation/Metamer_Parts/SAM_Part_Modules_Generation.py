"""
Generate 6 connected object-part masks per image using SAM.

Input:
  E:/#Stimsets/Metamer_Part_Demolish
Output:
  E:/#Stimsets/Metamer_Part_Modules

Notes:
- Prefers local SAM checkpoint; falls back to Hugging Face download.
- Designed for object images on near-uniform gray background.
"""
#%%
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
except Exception as exc:  # pragma: no cover - runtime dependency guard
    print(
        "Missing segment-anything dependency. "
        "Install with: pip install git+https://github.com/facebookresearch/segment-anything.git",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


INPUT_DIR = Path(r"E:\#Stimsets\Metamer_Demolish_v260508\Object_No_BK")
OUTPUT_DIR = Path(r"E:\#Stimsets\Metamer_Demolish_v260508\Metamer_Part_Modules")

TARGET_PARTS = 5
BG_GRAY = 127
FG_DIFF_THRESHOLD = 10
MIN_PART_AREA_RATIO = 0.02
MIN_OVERLAP_RATIO = 0.65
DETAIL_OPEN_KERNEL = 5
MAX_AREA_RATIO = 2.2
MAX_REBALANCE_STEPS = 48
MIN_FINAL_PART_FRACTION_OF_IDEAL = 0.55
MAX_MIN_AREA_FIX_STEPS = 64

SAM_MODEL_TYPE = "vit_h"
SAM_CHECKPOINT_LOCAL = Path(r"E:\#Stimsets\models\sam_vit_h_4b8939.pth")
SAM_CHECKPOINT_REPO = "facebook/sam-vit-huge"
SAM_CHECKPOINT_FILE = "sam_vit_h_4b8939.pth"
SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
SAM_CACHE_DIR = Path.home() / ".cache" / "segment_anything"

SAM_POINTS_PER_SIDE = 24
SAM_PRED_IOU_THRESH = 0.88
SAM_STABILITY_SCORE_THRESH = 0.95
SAM_CROP_N_LAYERS = 0
SAM_CROP_N_POINTS_DOWNSCALE_FACTOR = 2
SAM_MIN_MASK_REGION_AREA = 256

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def detect_foreground(rgb: np.ndarray) -> np.ndarray:
    """Detect foreground for near-gray-background object renders."""
    diff = np.max(np.abs(rgb.astype(np.int16) - int(BG_GRAY)), axis=2)
    fg = (diff > FG_DIFF_THRESHOLD).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats((fg > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return fg > 0
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    src = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
    if num <= 1:
        return src.astype(bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = labels == largest
    out_u8 = out.astype(np.uint8) * 255
    out_u8 = cv2.morphologyEx(out_u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return out_u8 > 0


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Fill internal holes while keeping the outer boundary unchanged."""
    src = (mask > 0).astype(np.uint8) * 255
    if src.size == 0:
        return mask.astype(bool)
    inv = cv2.bitwise_not(src)
    ff = inv.copy()
    h, w = ff.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(ff, flood_mask, seedPoint=(0, 0), newVal=0)
    holes = ff
    filled = cv2.bitwise_or(src, holes)
    return filled > 0


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    src = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
    comps: list[np.ndarray] = []
    for idx in range(1, num):
        if stats[idx, cv2.CC_STAT_AREA] <= 0:
            continue
        comps.append(labels == idx)
    return comps


def load_checkpoint() -> Path:
    if SAM_CHECKPOINT_LOCAL.is_file():
        return SAM_CHECKPOINT_LOCAL
    SAM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    hf_error: Exception | None = None
    try:
        downloaded = hf_hub_download(repo_id=SAM_CHECKPOINT_REPO, filename=SAM_CHECKPOINT_FILE)
        return Path(downloaded)
    except Exception as exc:
        hf_error = exc
        print(
            f"HF checkpoint download failed ({SAM_CHECKPOINT_REPO}/{SAM_CHECKPOINT_FILE}): {exc}",
            file=sys.stderr,
        )
        print("Falling back to direct SAM checkpoint URL...", file=sys.stderr)

    fallback_file = SAM_CACHE_DIR / SAM_CHECKPOINT_FILE
    if fallback_file.is_file():
        return fallback_file

    try:
        torch.hub.download_url_to_file(SAM_CHECKPOINT_URL, str(fallback_file), progress=True)
    except Exception as url_exc:
        raise RuntimeError(
            "Unable to download SAM checkpoint from both HF and direct URL.\n"
            f"HF error: {hf_error}\n"
            f"URL error: {url_exc}\n"
            "Please set SAM_CHECKPOINT_LOCAL to an existing .pth file."
        ) from url_exc
    return fallback_file


def build_sam_generator() -> SamAutomaticMaskGenerator:
    ckpt = load_checkpoint()
    model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(ckpt))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device=device)
    return SamAutomaticMaskGenerator(
        model=model,
        points_per_side=SAM_POINTS_PER_SIDE,
        pred_iou_thresh=SAM_PRED_IOU_THRESH,
        stability_score_thresh=SAM_STABILITY_SCORE_THRESH,
        crop_n_layers=SAM_CROP_N_LAYERS,
        crop_n_points_downscale_factor=SAM_CROP_N_POINTS_DOWNSCALE_FACTOR,
        min_mask_region_area=SAM_MIN_MASK_REGION_AREA,
    )


def select_candidate_masks(auto_masks: list[dict], fg_mask: np.ndarray) -> list[np.ndarray]:
    fg_area = int(fg_mask.sum())
    if fg_area == 0:
        return []
    min_pixels = max(64, int(fg_area * MIN_PART_AREA_RATIO))
    candidates: list[np.ndarray] = []
    for item in auto_masks:
        seg = item.get("segmentation")
        if seg is None:
            continue
        seg_mask = np.asarray(seg, dtype=bool) & fg_mask
        if DETAIL_OPEN_KERNEL >= 3:
            k = np.ones((DETAIL_OPEN_KERNEL, DETAIL_OPEN_KERNEL), np.uint8)
            seg_mask = cv2.morphologyEx((seg_mask.astype(np.uint8) * 255), cv2.MORPH_OPEN, k) > 0
        seg_area = int(seg_mask.sum())
        if seg_area < min_pixels:
            continue
        orig_area = int(np.asarray(seg, dtype=np.uint8).sum())
        if orig_area <= 0:
            continue
        overlap_ratio = seg_area / float(orig_area)
        if overlap_ratio < MIN_OVERLAP_RATIO:
            continue
        clean = keep_largest_component(seg_mask)
        if int(clean.sum()) >= min_pixels:
            candidates.append(clean)
    # Large-first assignment prevents tiny detail masks from grabbing pixels first.
    candidates.sort(key=lambda m: int(m.sum()), reverse=True)
    return candidates


def assign_uncovered_components(label_map: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    unlabeled = fg_mask & (label_map < 0)
    if not unlabeled.any():
        return label_map
    next_label = int(label_map.max()) + 1 if (label_map >= 0).any() else 0
    comps = connected_components(unlabeled)
    kernel = np.ones((3, 3), np.uint8)
    for comp in comps:
        comp_u8 = comp.astype(np.uint8)
        border = cv2.dilate(comp_u8, kernel, iterations=1).astype(bool) & (~comp)
        nbr = label_map[border]
        nbr = nbr[nbr >= 0]
        if nbr.size > 0:
            ids, cnt = np.unique(nbr, return_counts=True)
            best = int(ids[np.argmax(cnt)])
            label_map[comp] = best
        else:
            label_map[comp] = next_label
            next_label += 1
    return label_map


def build_initial_regions(candidates: list[np.ndarray], fg_mask: np.ndarray) -> np.ndarray:
    h, w = fg_mask.shape
    label_map = np.full((h, w), -1, dtype=np.int32)
    next_label = 0
    for mask in candidates:
        take = mask & fg_mask & (label_map < 0)
        if not take.any():
            continue
        for comp in connected_components(take):
            if comp.sum() < 32:
                continue
            label_map[comp] = next_label
            next_label += 1
    label_map = assign_uncovered_components(label_map, fg_mask)
    label_map[~fg_mask] = -1
    return label_map


def relabel_compact(label_map: np.ndarray) -> np.ndarray:
    out = np.full_like(label_map, -1)
    ids = [int(x) for x in np.unique(label_map) if x >= 0]
    for new_id, old_id in enumerate(ids):
        out[label_map == old_id] = new_id
    return out


def ensure_each_label_connected(label_map: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    label_map = relabel_compact(label_map)
    h, w = label_map.shape
    out = np.full((h, w), -1, dtype=np.int32)
    next_label = 0
    for rid in [int(x) for x in np.unique(label_map) if x >= 0]:
        region = label_map == rid
        comps = connected_components(region)
        comps.sort(key=lambda c: int(c.sum()), reverse=True)
        for comp in comps:
            out[comp] = next_label
            next_label += 1
    out = assign_uncovered_components(out, fg_mask)
    out[~fg_mask] = -1
    return relabel_compact(out)


def region_areas(label_map: np.ndarray) -> dict[int, int]:
    ids, cnt = np.unique(label_map[label_map >= 0], return_counts=True)
    return {int(i): int(c) for i, c in zip(ids, cnt)}


def build_neighbor_contacts(label_map: np.ndarray) -> dict[tuple[int, int], int]:
    contacts: dict[tuple[int, int], int] = {}
    for dy, dx in ((1, 0), (0, 1)):
        a = label_map[:-dy or None, :-dx or None]
        b = label_map[dy:, dx:]
        valid = (a >= 0) & (b >= 0) & (a != b)
        if not valid.any():
            continue
        pa = a[valid].astype(np.int64)
        pb = b[valid].astype(np.int64)
        lo = np.minimum(pa, pb)
        hi = np.maximum(pa, pb)
        pairs = np.stack([lo, hi], axis=1)
        uniq, cnt = np.unique(pairs, axis=0, return_counts=True)
        for (u, v), c in zip(uniq, cnt):
            key = (int(u), int(v))
            contacts[key] = contacts.get(key, 0) + int(c)
    return contacts


def merge_region_into_best_neighbor(label_map: np.ndarray, rid: int) -> np.ndarray:
    areas = region_areas(label_map)
    if len(areas) <= 1 or rid not in areas:
        return relabel_compact(label_map)
    contacts = build_neighbor_contacts(label_map)
    best_neighbor = None
    best_score = -1.0
    for (u, v), c in contacts.items():
        if rid not in (u, v):
            continue
        nbr = v if u == rid else u
        area_penalty = abs(areas[nbr] - areas[rid])
        score = float(c) - 0.002 * float(area_penalty)
        if score > best_score:
            best_score = score
            best_neighbor = nbr
    if best_neighbor is None:
        ids = [x for x in areas if x != rid]
        if not ids:
            return relabel_compact(label_map)
        best_neighbor = min(ids, key=lambda x: abs(areas[x] - areas[rid]))
    label_map = label_map.copy()
    label_map[label_map == rid] = int(best_neighbor)
    return relabel_compact(label_map)


def merge_smallest_once(label_map: np.ndarray) -> np.ndarray:
    areas = region_areas(label_map)
    if len(areas) <= 1:
        return label_map
    rid = min(areas, key=areas.get)
    return merge_region_into_best_neighbor(label_map, rid)


def split_largest_once(label_map: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    areas = region_areas(label_map)
    if not areas:
        return label_map
    rid = max(areas, key=areas.get)
    region = label_map == rid
    coords = np.argwhere(region)
    if coords.shape[0] < 64:
        return label_map

    yc = coords[:, 0].astype(np.float32)
    xc = coords[:, 1].astype(np.float32)
    pts = np.stack([xc, yc], axis=1)
    mean = pts.mean(axis=0, keepdims=True)
    centered = pts - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    proj = centered @ axis
    median = float(np.median(proj))
    side_a = proj <= median
    side_b = ~side_a
    if side_a.sum() < 20 or side_b.sum() < 20:
        q = float(np.quantile(proj, 0.4))
        side_a = proj <= q
        side_b = ~side_a
    if side_a.sum() < 20 or side_b.sum() < 20:
        return label_map

    a_mask = np.zeros_like(region, dtype=bool)
    b_mask = np.zeros_like(region, dtype=bool)
    a_pts = coords[side_a]
    b_pts = coords[side_b]
    a_mask[a_pts[:, 0], a_pts[:, 1]] = True
    b_mask[b_pts[:, 0], b_pts[:, 1]] = True

    a_mask = keep_largest_component(a_mask)
    b_mask = keep_largest_component(b_mask)
    covered = a_mask | b_mask
    remain = region & (~covered)
    if remain.any():
        a_dist = cv2.distanceTransform((~a_mask).astype(np.uint8), cv2.DIST_L2, 5)
        b_dist = cv2.distanceTransform((~b_mask).astype(np.uint8), cv2.DIST_L2, 5)
        to_a = remain & (a_dist <= b_dist)
        to_b = remain & (~to_a)
        a_mask[to_a] = True
        b_mask[to_b] = True
    if a_mask.sum() < 20 or b_mask.sum() < 20:
        return label_map

    out = label_map.copy()
    new_id = int(label_map.max()) + 1
    out[region] = -1
    out[a_mask] = rid
    out[b_mask] = new_id
    out = assign_uncovered_components(out, fg_mask)
    return relabel_compact(out)


def current_area_ratio(label_map: np.ndarray) -> float:
    areas = region_areas(label_map)
    if not areas:
        return float("inf")
    values = np.array(list(areas.values()), dtype=np.float32)
    min_area = float(values.min())
    max_area = float(values.max())
    if min_area <= 0.0:
        return float("inf")
    return max_area / min_area


def enforce_min_part_area(label_map: np.ndarray, fg_mask: np.ndarray, target_parts: int) -> np.ndarray:
    fg_area = int(fg_mask.sum())
    if fg_area <= 0 or target_parts <= 0:
        return relabel_compact(label_map)
    ideal = fg_area / float(target_parts)
    min_allowed = max(48, int(ideal * MIN_FINAL_PART_FRACTION_OF_IDEAL))

    label_map = relabel_compact(label_map)
    for _ in range(MAX_MIN_AREA_FIX_STEPS):
        areas = region_areas(label_map)
        if not areas:
            break
        tiny_ids = [rid for rid, area in areas.items() if area < min_allowed]
        if not tiny_ids:
            break

        tiny_id = min(tiny_ids, key=lambda rid: areas[rid])
        label_map = merge_region_into_best_neighbor(label_map, tiny_id)
        label_map = ensure_each_label_connected(label_map, fg_mask)

        # Keep the region count at target by splitting large regions back.
        split_guard = 0
        while len(region_areas(label_map)) < target_parts and split_guard < 8:
            before = len(region_areas(label_map))
            label_map = split_largest_once(label_map, fg_mask)
            label_map = ensure_each_label_connected(label_map, fg_mask)
            after = len(region_areas(label_map))
            split_guard += 1
            if after <= before:
                break

        while len(region_areas(label_map)) > target_parts:
            label_map = merge_smallest_once(label_map)
            label_map = ensure_each_label_connected(label_map, fg_mask)

    return relabel_compact(label_map)


def rebalance_area_with_fixed_parts(label_map: np.ndarray, fg_mask: np.ndarray, target_parts: int) -> np.ndarray:
    label_map = relabel_compact(label_map)
    for _ in range(MAX_REBALANCE_STEPS):
        n_regions = len(region_areas(label_map))
        if n_regions != target_parts:
            break
        ratio_before = current_area_ratio(label_map)
        if ratio_before <= MAX_AREA_RATIO:
            break

        before_map = label_map.copy()
        label_map = split_largest_once(label_map, fg_mask)
        if len(region_areas(label_map)) > target_parts:
            label_map = merge_smallest_once(label_map)
        label_map = ensure_each_label_connected(label_map, fg_mask)
        label_map = relabel_compact(label_map)

        ratio_after = current_area_ratio(label_map)
        if ratio_after >= ratio_before - 1e-3:
            label_map = before_map
            break
    return relabel_compact(label_map)


def force_exact_region_count(label_map: np.ndarray, fg_mask: np.ndarray, target_parts: int) -> np.ndarray:
    """Guarantee exact number of connected regions without dropping pixels."""
    label_map = ensure_each_label_connected(label_map, fg_mask)
    guard = 0
    while len(region_areas(label_map)) > target_parts and guard < 256:
        label_map = merge_smallest_once(label_map)
        label_map = ensure_each_label_connected(label_map, fg_mask)
        guard += 1
    while len(region_areas(label_map)) < target_parts and guard < 512:
        before = len(region_areas(label_map))
        label_map = split_largest_once(label_map, fg_mask)
        label_map = ensure_each_label_connected(label_map, fg_mask)
        after = len(region_areas(label_map))
        guard += 1
        if after <= before:
            break
    return relabel_compact(label_map)


def normalize_to_target_parts(label_map: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    label_map = ensure_each_label_connected(label_map, fg_mask)
    max_iter = 256
    for _ in range(max_iter):
        n_regions = len(region_areas(label_map))
        if n_regions == TARGET_PARTS:
            break
        if n_regions > TARGET_PARTS:
            label_map = merge_smallest_once(label_map)
        else:
            before = len(region_areas(label_map))
            label_map = split_largest_once(label_map, fg_mask)
            after = len(region_areas(label_map))
            if after == before:
                label_map = merge_smallest_once(label_map)
                label_map = split_largest_once(label_map, fg_mask)
        label_map = ensure_each_label_connected(label_map, fg_mask)
    label_map = enforce_min_part_area(label_map, fg_mask, TARGET_PARTS)
    label_map = rebalance_area_with_fixed_parts(label_map, fg_mask, TARGET_PARTS)
    label_map = force_exact_region_count(label_map, fg_mask, TARGET_PARTS)
    return relabel_compact(label_map)


def label_map_to_masks(label_map: np.ndarray, target_parts: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    ids = [int(x) for x in np.unique(label_map) if x >= 0]
    ids = sorted(ids)
    if len(ids) == 0:
        return masks
    if len(ids) != target_parts:
        raise RuntimeError(
            f"label_map contains {len(ids)} regions, expected exactly {target_parts}. "
            "Use force_exact_region_count before exporting masks."
        )
    for rid in ids:
        masks.append(fill_mask_holes(label_map == rid))
    return masks


def make_visualization(rgb: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    vis = rgb.astype(np.float32).copy()
    palette = np.array(
        [
            [230, 25, 75],
            [60, 180, 75],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
        ],
        dtype=np.float32,
    )
    alpha = 0.45
    for idx, mask in enumerate(masks):
        if not mask.any():
            continue
        color = palette[idx % len(palette)]
        vis[mask] = vis[mask] * (1.0 - alpha) + color * alpha
        ys, xs = np.where(mask)
        cy = int(np.median(ys))
        cx = int(np.median(xs))
        cv2.putText(
            vis,
            str(idx + 1),
            (max(0, cx - 8), max(12, cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return np.clip(vis, 0, 255).astype(np.uint8)


def save_outputs(stem: str, rgb: np.ndarray, masks: list[np.ndarray], out_root: Path) -> None:
    item_dir = out_root / stem
    item_dir.mkdir(parents=True, exist_ok=True)
    for idx, mask in enumerate(masks, start=1):
        out = (mask.astype(np.uint8) * 255)
        Image.fromarray(out, mode="L").save(item_dir / f"mask_{idx:02d}.png")
    vis = make_visualization(rgb, masks)
    Image.fromarray(vis, mode="RGB").save(item_dir / f"{stem}_vis.png")


def process_image(path: Path, generator: SamAutomaticMaskGenerator) -> tuple[bool, str]:
    img = Image.open(path).convert("RGB")
    rgb = np.asarray(img)
    fg = detect_foreground(rgb)
    if int(fg.sum()) < 128:
        return False, f"{path.name}: foreground too small"

    auto_masks = generator.generate(rgb)
    candidates = select_candidate_masks(auto_masks, fg)
    if not candidates:
        candidates = [fg]

    label_map = build_initial_regions(candidates, fg)
    label_map = normalize_to_target_parts(label_map, fg)
    masks = label_map_to_masks(label_map, TARGET_PARTS)
    if len(masks) != TARGET_PARTS:
        return False, f"{path.name}: failed to reach {TARGET_PARTS} parts"

    save_outputs(path.stem, rgb, masks, OUTPUT_DIR)
    ratio = current_area_ratio(label_map)
    return True, f"{path.name}: ok ({len(candidates)} candidates, max/min={ratio:.2f})"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images = list_images(INPUT_DIR)
    if not images:
        print(f"No images found in {INPUT_DIR}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Loading SAM model ({SAM_MODEL_TYPE})...")
    generator = build_sam_generator()
    ok_count = 0
    fail_count = 0

    for path in tqdm(images, desc="SAM 6-part"):
        try:
            ok, msg = process_image(path, generator)
            if ok:
                ok_count += 1
            else:
                fail_count += 1
                print(msg, file=sys.stderr)
        except Exception as exc:
            fail_count += 1
            print(f"{path.name}: {exc}", file=sys.stderr)

    print(f"Done. success={ok_count}, failed={fail_count}, output={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
