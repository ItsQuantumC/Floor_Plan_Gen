import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image

from models.generator import UNetGenerator
from models.discriminator import PatchDiscriminator
from dataset import FloorPlanDataset  # same dataset as before

# ---- CONFIG ----
BATCH_SIZE = 4
IMAGE_SIZE = 256
EPOCHS = 100
LAMBDA_L1 = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "outputs_optimized"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---- TRANSFORM ----
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ---- LOAD DATASET ----
train_ds = FloorPlanDataset("data/inputs", "data/targets", transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# ---- INIT MODELS ----
gen = UNetGenerator().to(DEVICE)
disc = PatchDiscriminator().to(DEVICE)

# ---- LOSS + OPTIM ----
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()
opt_G = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))

# ---- TRAIN LOOP ----
for epoch in range(EPOCHS):
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)

        # ---- Add Gaussian Noise to Input ----
        noise = torch.randn_like(x) * 0.05
        x_noisy = torch.clamp(x + noise, -1, 1)

        fake_y = gen(x_noisy)

        # ---- Label Smoothing ----
        real_label = torch.ones_like(disc(x, y)) * 0.9  # 0.9 instead of 1
        fake_label = torch.zeros_like(disc(x, fake_y))

        # ---- Train Discriminator ----
        D_real = disc(x, y)
        D_fake = disc(x, fake_y.detach())
        loss_D_real = criterion_GAN(D_real, real_label)
        loss_D_fake = criterion_GAN(D_fake, fake_label)
        loss_D = 0.5 * (loss_D_real + loss_D_fake)

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # ---- Train Generator ----
        D_fake_for_G = disc(x, fake_y)
        loss_G_GAN = criterion_GAN(D_fake_for_G, torch.ones_like(D_fake_for_G))  # maximizing D(x, G(x))
        loss_G_L1 = criterion_L1(fake_y, y)
        loss_G = loss_G_GAN + LAMBDA_L1 * loss_G_L1

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        # ---- Logging ----
        if i % 50 == 0:
            print(f"Epoch [{epoch}/{EPOCHS}] Step [{i}/{len(train_loader)}] "
                  f"Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}")

    # ---- Save Samples Every Epoch ----
    save_image((fake_y * 0.5 + 0.5), f"{SAVE_DIR}/fake_{epoch}.png")
    save_image((y * 0.5 + 0.5), f"{SAVE_DIR}/real_{epoch}.png")
    save_image((x * 0.5 + 0.5), f"{SAVE_DIR}/input_{epoch}.png")

    # ---- Save Checkpoints ----
    torch.save(gen.state_dict(), f"{SAVE_DIR}/gen_epoch{epoch}.pth")
    torch.save(disc.state_dict(), f"{SAVE_DIR}/disc_epoch{epoch}.pth")


