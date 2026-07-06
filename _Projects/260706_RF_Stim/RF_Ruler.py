'''
生成用于测感受野的刺激。

屏幕: 600×600 pix = 15°×15°
FOB 图在 3°×3° (5×5 栅格) 与 6°×6° (3×3 栅格) 两种尺寸下放置。
命名: 0000/0001/0002 为纯黑/纯白/纯灰; XYrc.jpg 中 X=FOB序号, Y=尺寸(1=3°,2=6°), rc=行列(左上角为11)。
'''

#%%
from pathlib import Path

import numpy as np
from PIL import Image

fob_path = r'E:\#Stimsets\RF_Ruler_v260706\fob_example'
savepath = r'E:\#Stimsets\RF_Ruler_v260706'

CANVAS_PX = 600
FIELD_DEG = 15.0
GRAY = 127
PX_PER_DEG = CANVAS_PX / FIELD_DEG

FOB_NAMES = ('Body', 'Face', 'Object')
SIZE_CONFIG = {
    1: {'stim_deg': 3.0, 'grid': 5},
    2: {'stim_deg': 6.0, 'grid': 3},
}


def deg_to_px(deg: float) -> int:
    return int(round(deg * PX_PER_DEG))


def grid_top_left_px(row: int, col: int, stim_deg: float, grid_n: int) -> tuple[int, int]:
    """row/col 从 1 开始; 左上角栅格为 (1,1), 且栅格覆盖整个 15° 视场并包含屏幕中心。"""
    stim_px = deg_to_px(stim_deg)
    step_deg = (FIELD_DEG - stim_deg) / (grid_n - 1)
    step_px = deg_to_px(step_deg)
    x0 = (col - 1) * step_px
    y0 = (row - 1) * step_px
    return x0, y0


def make_canvas() -> np.ndarray:
    return np.full((CANVAS_PX, CANVAS_PX, 3), GRAY, dtype=np.uint8)


def paste_stim(canvas: np.ndarray, stim_rgb: np.ndarray, row: int, col: int,
               stim_deg: float, grid_n: int) -> np.ndarray:
    stim_px = deg_to_px(stim_deg)
    stim_img = Image.fromarray(stim_rgb).resize((stim_px, stim_px), Image.Resampling.LANCZOS)
    stim_arr = np.asarray(stim_img)
    x0, y0 = grid_top_left_px(row, col, stim_deg, grid_n)
    x1 = min(x0 + stim_px, CANVAS_PX)
    y1 = min(y0 + stim_px, CANVAS_PX)
    src_x0 = max(0, -x0)
    src_y0 = max(0, -y0)
    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)
    canvas[dst_y0:y1, dst_x0:x1] = stim_arr[src_y0:src_y0 + (y1 - dst_y0),
                                                src_x0:src_x0 + (x1 - dst_x0)]
    return canvas


def load_fob_images(fob_dir: Path) -> list[np.ndarray]:
    """按 Body / Face / Object 顺序加载三张 FOB 图。

    优先匹配文件名含 Body/Face/Object; 否则按文件名排序取目录内前 3 张图。
    """
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
    images = []
    named_paths: list[Path] = []
    for name in FOB_NAMES:
        matches: list[Path] = []
        for ext in exts:
            matches.extend(fob_dir.glob(f'{name}{ext[1:]}'))
            matches.extend(fob_dir.glob(f'{name.lower()}{ext[1:]}'))
        matches = sorted(set(matches))
        if matches:
            named_paths.append(matches[0])

    if len(named_paths) == len(FOB_NAMES):
        paths = named_paths
    else:
        all_paths: list[Path] = []
        for ext in exts:
            all_paths.extend(fob_dir.glob(ext))
        paths = sorted(set(all_paths))
        if len(paths) < len(FOB_NAMES):
            raise FileNotFoundError(
                f'在 {fob_dir} 中需要至少 {len(FOB_NAMES)} 张图, 当前仅找到 {len(paths)} 张'
            )
        paths = paths[: len(FOB_NAMES)]

    for fob_idx, path in enumerate(paths, start=1):
        img = Image.open(path).convert('RGB')
        images.append(np.asarray(img))
        print(f'  FOB{fob_idx} ({FOB_NAMES[fob_idx - 1]}): {path.name}')
    return images


def save_image(arr: np.ndarray, out_path: Path) -> None:
    Image.fromarray(arr).save(out_path, quality=95)


def generate_solid_colors(out_dir: Path) -> None:
    colors = [(0, 0, 0), (255, 255, 255), (GRAY, GRAY, GRAY)]
    labels = ('纯黑', '纯白', f'纯灰({GRAY})')
    for idx, (color, label) in enumerate(zip(colors, labels)):
        arr = np.full((CANVAS_PX, CANVAS_PX, 3), color, dtype=np.uint8)
        fname = f'{idx:04d}.jpg'
        save_image(arr, out_dir / fname)
        print(f'  {fname}  {label}')


def generate_fob_stimuli(fob_images: list[np.ndarray], out_dir: Path) -> None:
    for fob_idx, stim_rgb in enumerate(fob_images, start=1):
        for size_code, cfg in SIZE_CONFIG.items():
            grid_n = cfg['grid']
            stim_deg = cfg['stim_deg']
            for row in range(1, grid_n + 1):
                for col in range(1, grid_n + 1):
                    canvas = make_canvas()
                    canvas = paste_stim(canvas, stim_rgb, row, col, stim_deg, grid_n)
                    fname = f'{fob_idx}{size_code}{row}{col}.jpg'
                    save_image(canvas, out_dir / fname)


def loc_fob(size_code: int, row: int, col: int) -> str:
  return f'LOC_{size_code}{row}{col}'


def generate_tsv(out_dir: Path) -> Path:
    """生成与 sti150_info.tsv 相同格式的刺激信息表。"""
    rows: list[tuple[int, str, str, str]] = []
    idx = 1

    for fname in ('0000.jpg', '0001.jpg', '0002.jpg'):
        rows.append((idx, fname, 'RF', 'CTR'))
        idx += 1

    for fob_idx in range(1, len(FOB_NAMES) + 1):
        for size_code, cfg in SIZE_CONFIG.items():
            grid_n = cfg['grid']
            for row in range(1, grid_n + 1):
                for col in range(1, grid_n + 1):
                    fname = f'{fob_idx}{size_code}{row}{col}.jpg'
                    rows.append((idx, fname, 'RF', loc_fob(size_code, row, col)))
                    idx += 1

    tsv_path = out_dir / 'RF_Ruler_info.tsv'
    with tsv_path.open('w', encoding='utf-8', newline='') as f:
        f.write('Index\tFileName\tCategory\tFOB\n')
        for row_idx, fname, category, fob in rows:
            f.write(f'{row_idx}\t{fname}\t{category}\t{fob}\n')
    return tsv_path


def main() -> None:
    out_dir = Path(savepath)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('生成纯色刺激...')
    generate_solid_colors(out_dir)

    print('加载 FOB 图片...')
    fob_images = load_fob_images(Path(fob_path))

    print('生成 FOB 栅格刺激...')
    generate_fob_stimuli(fob_images, out_dir)

    print('生成 TSV 信息表...')
    tsv_path = generate_tsv(out_dir)
    print(f'  {tsv_path.name}')

    n_solid = 3
    n_fob = len(FOB_NAMES) * sum(cfg['grid'] ** 2 for cfg in SIZE_CONFIG.values())
    print(f'完成: 共 {n_solid + n_fob} 张图片 -> {out_dir}')

#%%
if __name__ == '__main__':
    main()
