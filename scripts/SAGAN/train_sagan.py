import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from generator_sagan import Generator
from discriminator_sagan import Discriminator
from dataset_sagan import FloorPlanImages

def main():
    # ---- CONFIG ----
    z_dim      = 128
    img_size   = 128
    batch_size = 16
    epochs     = 100
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir   = "outputs_sagan"
    os.makedirs(save_dir, exist_ok=True)

    # ---- DATA ----
    dataset = FloorPlanImages("data/colored_targets", img_size=img_size)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, num_workers=0              # set to 0 to avoid spawn issues
    )

    # ---- MODELS, LOSS, OPT ----
    G = Generator(z_dim=z_dim).to(device)
    D = Discriminator().to(device)
    criterion = nn.BCEWithLogitsLoss()
    opt_G     = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5,0.999))
    opt_D     = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5,0.999))
    fixed_noise = torch.randn(16, z_dim, device=device)

    # ---- TRAIN LOOP ----
    for epoch in range(epochs):
        for real in loader:
            real = real.to(device)
            b_size = real.size(0)

            # Train D
            noise = torch.randn(b_size, z_dim, device=device)
            fake  = G(noise)
            D_real = D(real).squeeze()
            D_fake = D(fake.detach()).squeeze()
            loss_D = 0.5*(criterion(D_real, torch.ones_like(D_real)*0.9)
                       + criterion(D_fake, torch.zeros_like(D_fake)))
            D.zero_grad(); loss_D.backward(); opt_D.step()

            # Train G
            D_fake_for_G = D(fake).squeeze()
            loss_G = criterion(D_fake_for_G, torch.ones_like(D_fake_for_G))
            G.zero_grad(); loss_G.backward(); opt_G.step()

        print(f"[{epoch+1}/{epochs}] Loss_D: {loss_D:.4f} Loss_G: {loss_G:.4f}")

        if (epoch+1)%10==0 or epoch==0:
            with torch.no_grad():
                samples = G(fixed_noise).cpu()
                save_image((samples*0.5+0.5),
                           f"{save_dir}/fixed_epoch{epoch+1:03d}.png", nrow=4)

        if (epoch+1)%50==0:
            torch.save(G.state_dict(), os.path.join(save_dir, f"G_{epoch+1}.pth"))
            torch.save(D.state_dict(), os.path.join(save_dir, f"D_{epoch+1}.pth"))

if __name__ == "__main__":
    main()
