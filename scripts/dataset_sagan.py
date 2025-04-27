import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class FloorPlanImages(Dataset):
    def __init__(self, img_dir, img_size=128):
        self.files = sorted(os.listdir(img_dir))
        self.img_dir = img_dir
        self.tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.files[idx])).convert("RGB")
        return self.tf(img)
