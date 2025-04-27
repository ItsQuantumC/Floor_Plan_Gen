import os
import shutil

# 1) Point this at your cubicasa5k/colorful directory
SOURCE_ROOT = os.path.join(os.path.dirname(__file__), "..", "cubicasa5k", "colorful")
# 2) Where you want all the .png’s to go
TARGET_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "colored_targets")

os.makedirs(TARGET_DIR, exist_ok=True)

copied = 0
for dirpath, dirnames, filenames in os.walk(SOURCE_ROOT):
    # dirpath might be .../cubicasa5k/colorful/158, /34, /220, etc.
    sub = os.path.relpath(dirpath, SOURCE_ROOT)  # e.g. "158" or "34" or "."
    prefix = "" if sub == "." else sub + "_"

    for fn in filenames:
        if fn.lower().endswith(".png"):
            src = os.path.join(dirpath, fn)
            # prefix the subfolder name to avoid collisions
            dst_name = prefix + fn
            dst = os.path.join(TARGET_DIR, dst_name)
            shutil.copy2(src, dst)
            copied += 1

print(f"✔ Found and copied {copied} PNG files into\n   {TARGET_DIR}")
