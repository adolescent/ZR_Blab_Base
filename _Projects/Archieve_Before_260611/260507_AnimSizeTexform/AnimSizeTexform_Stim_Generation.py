#%%
from pathlib import Path

from PIL import Image

fob_folder = r"E:\#Stimsets\AnimSizeTexform_v260507\FOB72"
save_folder = r"E:\#Stimsets\AnimSizeTexform_v260507\_Overall"
fig_root_1 = r"G:\我的云端硬盘\#BLab_Works\_260428_Ani_Texform\AnimSizeTexform"
fig_root_2 = r"G:\我的云端硬盘\#BLab_Works\_260428_Ani_Texform\AnimSizeTexformHighContrast"

TARGET_SIZE = (400, 400)


def list_png_files(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"],
        key=lambda p: p.name.lower(),
    )


def resize_and_save(src_paths: list[Path], dst_folder: Path, start_index: int) -> int:
    current_index = start_index
    for src in src_paths:
        dst_name = f"{current_index:05d}.png"
        dst_path = dst_folder / dst_name
        with Image.open(src) as img:
            resized = img.convert("RGB").resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            resized.save(dst_path, format="PNG")
        current_index += 1
    return current_index


def get_prefix_subfolders(root: Path, prefix: str, limit: int = 4) -> list[Path]:
    matched = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith(prefix.lower())],
        key=lambda p: p.name.lower(),
    )
    return matched[:limit]


def collect_pngs_from_subfolders(subfolders: list[Path]) -> list[Path]:
    all_pngs: list[Path] = []
    for folder in subfolders:
        all_pngs.extend(list_png_files(folder))
    return all_pngs


def main() -> None:
    fob_path = Path(fob_folder)
    out_path = Path(save_folder)
    root1 = Path(fig_root_1)
    root2 = Path(fig_root_2)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1) FOB_folder -> 00001, 00002, ...
    fob_pngs = list_png_files(fob_path)
    next_id = resize_and_save(fob_pngs, out_path, start_index=1)
    print(f"FOB done: {len(fob_pngs)} files, next id {next_id:05d}")

    # 2) fig_root_1: Orig* (first 4) -> 10001, 10002, ...
    root1_orig_subfolders = get_prefix_subfolders(root1, "Orig", limit=4)
    root1_orig_pngs = collect_pngs_from_subfolders(root1_orig_subfolders)
    next_id = resize_and_save(root1_orig_pngs, out_path, start_index=10001)
    print(
        f"fig_root_1 Orig done: {len(root1_orig_pngs)} files from {len(root1_orig_subfolders)} folders, "
        f"next id {next_id:05d}"
    )

    # 3) fig_root_1: Texform* (first 4) -> 20001, 20002, ...
    root1_tex_subfolders = get_prefix_subfolders(root1, "Texform", limit=4)
    root1_tex_pngs = collect_pngs_from_subfolders(root1_tex_subfolders)
    next_id = resize_and_save(root1_tex_pngs, out_path, start_index=20001)
    print(
        f"fig_root_1 Texform done: {len(root1_tex_pngs)} files from {len(root1_tex_subfolders)} folders, "
        f"next id {next_id:05d}"
    )

    # 4) fig_root_2 with same structure -> 30001 / 40001
    root2_orig_subfolders = get_prefix_subfolders(root2, "Orig", limit=4)
    root2_orig_pngs = collect_pngs_from_subfolders(root2_orig_subfolders)
    next_id = resize_and_save(root2_orig_pngs, out_path, start_index=30001)
    print(
        f"fig_root_2 Orig done: {len(root2_orig_pngs)} files from {len(root2_orig_subfolders)} folders, "
        f"next id {next_id:05d}"
    )

    root2_tex_subfolders = get_prefix_subfolders(root2, "Texform", limit=4)
    root2_tex_pngs = collect_pngs_from_subfolders(root2_tex_subfolders)
    next_id = resize_and_save(root2_tex_pngs, out_path, start_index=40001)
    print(
        f"fig_root_2 Texform done: {len(root2_tex_pngs)} files from {len(root2_tex_subfolders)} folders, "
        f"next id {next_id:05d}"
    )

    print("All done.")


if __name__ == "__main__":
    main()
