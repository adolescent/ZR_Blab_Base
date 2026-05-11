import random
from pathlib import Path

from PIL import Image


def main() -> None:
    source_dir = Path(r"Z:\Monkey\Stimuli\LYP\NSD1000")
    output_path = Path(
        r"C:\#working_folder\#Codes\ZR_Blab_Base\_Projects\260421_Odd_One_Decode\nsd_random_10s.gif"
    )

    bmp_files = list(source_dir.glob("*.bmp"))
    if not bmp_files:
        raise FileNotFoundError(f"No BMP files found in: {source_dir}")

    total_ms = 10_000
    show_ms = 300
    blank_ms = 300

    first = Image.open(bmp_files[0]).convert("RGB")
    width, height = first.size
    gray_frame = Image.new("RGB", (width, height), (128, 128, 128))

    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)

    frames: list[Image.Image] = []
    durations: list[int] = []
    elapsed = 0

    while elapsed < total_ms:
        img = Image.open(random.choice(bmp_files)).convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), resampling)

        d_show = min(show_ms, total_ms - elapsed)
        if d_show <= 0:
            break
        frames.append(img)
        durations.append(d_show)
        elapsed += d_show

        if elapsed >= total_ms:
            break

        d_blank = min(blank_ms, total_ms - elapsed)
        if d_blank <= 0:
            break
        frames.append(gray_frame.copy())
        durations.append(d_blank)
        elapsed += d_blank

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )

    print(f"Saved: {output_path}")
    print(f"Total frames: {len(frames)}")
    print(f"Total duration (ms): {sum(durations)}")


if __name__ == "__main__":
    main()
