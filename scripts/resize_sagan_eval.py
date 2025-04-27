# scripts/prepare_real_for_fid.py

import os
from PIL import Image

SRC_DIR  = "data/colored_targets"
DST_DIR  = "data/colored_targets_128"
IMG_SIZE = 128

os.makedirs(DST_DIR, exist_ok=True)

for fn in os.listdir(SRC_DIR):
    if not fn.lower().endswith((".png","jpg","jpeg")):
        continue
    img = Image.open(os.path.join(SRC_DIR, fn)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    img.save(os.path.join(DST_DIR, fn))

print(f"Resized {len(os.listdir(DST_DIR))} images into {DST_DIR}")
