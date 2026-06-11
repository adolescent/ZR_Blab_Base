#%%
import OS_Tools as ot
import os
from PIL import Image


metamer_paths = ot.Get_File_Name(r'E:\#Stimsets\Metamer_P4_C4321_Object_STI150_1300','.jpg')[:1000]
fob_paths = ot.Get_File_Name(r'Z:\Monkey\Stimuli\ZR\FOB72','.png')
nsd_files = ot.Get_File_Name(r'E:\#Stimsets\NSD1000','.bmp')

fob_paths.sort()
nsd_files.sort()
metamer_paths.sort()

savepath = r'E:\#Stimsets\Metamer_NSD_FOB_v260420'
target_size = (400, 400)

#%%
os.makedirs(savepath, exist_ok=True)


def save_as_jpg(src_path, dst_path, quality=95, subsampling=None):
    with Image.open(src_path) as img:
        # JPEG does not support alpha channel; convert safely to RGB.
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # Unify all stimuli to the same spatial size.
        resample_filter = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        img = img.resize(target_size, resample=resample_filter)

        save_kwargs = {"format": "JPEG", "quality": quality}
        if subsampling is not None:
            save_kwargs["subsampling"] = subsampling

        img.save(dst_path, **save_kwargs)


saved_count = 0

# 1) FOB files repeated 3 times: 00001.jpg, 00002.jpg, ...
for repeat_idx in range(3):
    for idx, src_file in enumerate(fob_paths, start=1):
        out_idx = repeat_idx * len(fob_paths) + idx
        out_name = f"{out_idx:05d}.jpg"
        out_path = os.path.join(savepath, out_name)
        save_as_jpg(src_file, out_path, quality=95)
        saved_count += 1

# 2) Metamer files: 10001.jpg, 10002.jpg, ...
for idx, src_file in enumerate(metamer_paths, start=1):
    out_name = f"{10000 + idx:05d}.jpg"
    out_path = os.path.join(savepath, out_name)
    save_as_jpg(src_file, out_path, quality=95)
    saved_count += 1

# 3) NSD files: 50001.jpg, 50002.jpg, ... with higher JPEG quality.
for idx, src_file in enumerate(nsd_files, start=1):
    out_name = f"{50000 + idx:05d}.jpg"
    out_path = os.path.join(savepath, out_name)
    save_as_jpg(src_file, out_path, quality=100, subsampling=0)
    saved_count += 1

print(f"FOB: {len(fob_paths) * 3}, Metamer: {len(metamer_paths)}, NSD: {len(nsd_files)}")
print(f"Total saved images: {saved_count}")
