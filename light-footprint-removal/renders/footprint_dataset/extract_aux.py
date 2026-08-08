# extract_aux.py — turns the mask_raw renders (RGBA PNGs with transparent
# background) into clean binary mask PNGs.
#
# Run OUTSIDE Blender, with plain Python (locally or in Colab):
#   pip install numpy pillow
#   python extract_aux.py ~/footprint_dataset/with
#
# (argument = the mode folder that contains the 'mask_raw' subfolder)

import sys
import os
import glob
import numpy as np
from PIL import Image


def main(mode_dir):
    raw_files = sorted(glob.glob(os.path.join(mode_dir, "mask_raw", "*.png")))
    if not raw_files:
        sys.exit(f"No PNGs found in {mode_dir}/mask_raw")

    mask_dir = os.path.join(mode_dir, "mask")
    os.makedirs(mask_dir, exist_ok=True)

    for path in raw_files:
        rgba = np.array(Image.open(path).convert("RGBA"))
        alpha = rgba[..., 3]
        mask = (alpha > 127).astype(np.uint8) * 255
        Image.fromarray(mask, mode="L").save(
            os.path.join(mask_dir, os.path.basename(path)))

    print(f"Wrote {len(raw_files)} masks -> {mask_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python extract_aux.py <path-to-mode-folder, e.g. .../with>")
    main(sys.argv[1])
