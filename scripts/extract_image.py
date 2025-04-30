import os
import shutil

SOURCE_ROOT = os.path.join(os.path.dirname(__file__), "..", "cubicasa5k", "colorful")

TARGET_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "colored_targets")

os.makedirs(TARGET_DIR, exist_ok=True)

copied = 0
for dirpath, dirnames, filenames in os.walk(SOURCE_ROOT):
   
    sub = os.path.relpath(dirpath, SOURCE_ROOT)  
    prefix = "" if sub == "." else sub + "_"

    for fn in filenames:
        if fn.lower().endswith(".png"):
            src = os.path.join(dirpath, fn)
            # prefix subfolder name to avoid collisions
            dst_name = prefix + fn
            dst = os.path.join(TARGET_DIR, dst_name)
            shutil.copy2(src, dst)
            copied += 1

print(f"✔ Found and copied {copied} PNG files into\n   {TARGET_DIR}")
