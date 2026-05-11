"""Post-process SFM AVIs: vflip, pad to 1918x1078 (gray 127), white corner box."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

raw_path = Path(r"E:\#Stimsets\Depth_Cue_Invariant\tool_faces_bodies_chairs_depth_cues\SFM")
save_path = Path(r"E:\#Stimsets\Depth_Cue_Invariant\tool_faces_bodies_chairs_depth_cues\SFM_adj")


def main() -> None:
    save_path.mkdir(parents=True, exist_ok=True)
    avi_files = sorted(raw_path.glob("*.avi"))
    if not avi_files:
        print(f"No .avi files under {raw_path}", file=sys.stderr)
        sys.exit(1)
    total = len(avi_files)
    filter_chain = (
        "vflip,"
        "pad=1918:1078:(ow-iw)/2:(oh-ih)/2:color=0x7f7f7f,"
        "drawbox=x=0:y=ih-50:w=50:h=50:color=white:t=fill"
    )
    for i, src in enumerate(avi_files, start=1):
        dst = save_path / f"{src.stem}.avi"
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-vf",
            filter_chain,
            "-c:v",
            "mpeg4",
            "-q:v",
            "1",
            "-r",
            "30",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(dst),
        ]
        print(f"[{i}/{total}] {src.name} -> {dst.name}")
        subprocess.run(cmd, check=True)
    print(f"Done: {total} files -> {save_path}")


if __name__ == "__main__":
    main()
