import os
from PIL import Image
from torch.utils.data import Dataset

class FloorPlanDataset(Dataset):
    def __init__(self, inp_dir, tgt_dir, transform):
        self.inp_dir = inp_dir
        self.tgt_dir = tgt_dir
        self.transform = transform
        self.files = sorted(os.listdir(inp_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        x = Image.open(os.path.join(self.inp_dir, file)).convert("RGB")
        y = Image.open(os.path.join(self.tgt_dir, file)).convert("RGB")
        return self.transform(x), self.transform(y)
