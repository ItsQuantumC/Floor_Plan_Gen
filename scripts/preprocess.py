import os, cv2
from glob import glob

raw_dir = "data/raw"
inp_dir = "data/inputs"
tgt_dir = "data/targets"

os.makedirs(inp_dir, exist_ok=True)
os.makedirs(tgt_dir, exist_ok=True)

H, W = 256, 256

for path in glob(os.path.join(raw_dir, "*.jpg")):
    img = cv2.imread(path)
    img = cv2.resize(img, (W, H))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    sketch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    fn = os.path.basename(path)
    cv2.imwrite(os.path.join(inp_dir, fn), sketch)
    cv2.imwrite(os.path.join(tgt_dir, fn), img)
