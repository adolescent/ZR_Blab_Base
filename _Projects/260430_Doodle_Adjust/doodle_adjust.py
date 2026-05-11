#%%
# 
import os
import shutil

path1 = r"Z:\Monkey\Stimuli\ZR\Doodle_AI_v260121"
fob_path = r"Z:\Monkey\Stimuli\YJ\WordLocalizer"
savepath = r"Z:\Monkey\Stimuli\ZR\Doodle_AI_v260430"


def copy_item(src_path: str, dst_path: str) -> None:
    """Copy file or folder without modifying source."""
    if os.path.isdir(src_path):
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    else:
        shutil.copy2(src_path, dst_path)


def copy_non_zero_from_path1() -> None:
    for name in os.listdir(path1):
        if not name or name.startswith("0"):
            continue
        src = os.path.join(path1, name)
        dst = os.path.join(savepath, name)
        copy_item(src, dst)
        print(f"[path1] copied: {name}")


def copy_fob_as_zero_jpg() -> None:
    files = sorted(
        [
            n
            for n in os.listdir(fob_path)
            if os.path.isfile(os.path.join(fob_path, n))
            and os.path.splitext(n)[1].lower() in {".jpg", ".jpeg"}
        ],
        key=str.lower,
    )
    width = max(4, len(str(len(files))))

    for idx, name in enumerate(files, start=1):
        src = os.path.join(fob_path, name)
        new_name = f"{idx:0{width}d}.jpg"
        dst = os.path.join(savepath, new_name)
        shutil.copy2(src, dst)
        print(f"[fob_path] copied: {name} -> {new_name}")

#%%
if __name__ == "__main__":
    os.makedirs(savepath, exist_ok=True)
    copy_non_zero_from_path1()
    copy_fob_as_zero_jpg()
    print("Done.")
