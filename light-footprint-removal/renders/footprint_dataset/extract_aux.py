# extract_aux.py — thresholds the alpha channel of the mask_raw renders into
# binary mask PNGs. Plain Python, no Blender:
#   python extract_aux.py ~/footprint_dataset/with

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
