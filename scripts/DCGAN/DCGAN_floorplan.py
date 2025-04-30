# Revamped DCGAN for Floor Plan Generation with Anti-Mode Collapse Strategies
import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from torch.nn.utils import spectral_norm

# Hyperparameters
latent_dim = 100
image_size = 128
batch_size = 64
channels_img = 1  # Change to 3 if using RGB
epochs = 100
lr = 2e-4
beta1 = 0.5
noise_std = 0.05
smooth_real = 0.9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transform
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.Grayscale(num_output_channels=1),  # Change to RGB transform if using 3 channels
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Dataset and loader
dataset = datasets.ImageFolder(root="data/train", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

""" Architecture guidelines for stable Deep Convolutional GANs
        •Replace any pooling layers with strided convolutions (discriminator) and fractional-strided
        convolutions (generator).
        •Use batchnorm in both the generator and the discriminator.
        •Remove fully connected hidden layers for deeper architectures.
        •Use ReLU activation in generator for all layers except for the output, which uses Tanh.
        •Use LeakyReLU activation in the discriminator for all layers.

        ^^^ as mentioned in the paper
 """


# Minibatch Discrimination
class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims):
        super().__init__()
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.T = nn.Parameter(torch.Tensor(in_features, out_features * kernel_dims))
        nn.init.normal_(self.T, 0, 1)

    def forward(self, x):
        M = x.matmul(self.T)
        M = M.view(-1, self.out_features, self.kernel_dims)
        out_tensor = []
        for i in range(M.size(0)):
            out_i = torch.sum(torch.exp(-torch.sum((M[i].unsqueeze(0) - M)**2, dim=2)), dim=0) - 1
            out_tensor.append(out_i)
        out = torch.stack(out_tensor)
        return torch.cat([x, out], dim=1)

# Generator with LeakyRELU activation
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, channels_img, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# Discriminator with Spectral Normalization + Minibatch Discrimination
class Discriminator(nn.Module):
    def __init__(self, channels_img):
        super().__init__()
        self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(channels_img, 64, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(512, 1024, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.minibatch = MinibatchDiscrimination(1024 * 4 * 4, 100, 5)
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 4 * 4 + 100, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.minibatch(x)
        return self.classifier(x).view(-1)

# Initialize
gen = Generator(latent_dim, channels_img).to(device)
disc = Discriminator(channels_img).to(device)
criterion = nn.BCELoss()
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta1, 0.999))
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta1, 0.999))

# Training loop
for epoch in range(epochs):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        noise += 0.05 * torch.randn_like(noise)  # Latent noise variation

        # Generate fake images
        fake = gen(noise)
        fake_for_disc = fake.detach().clone()
        real_for_disc = real.clone()

        real_for_disc += noise_std * torch.randn_like(real_for_disc)
        fake_for_disc += noise_std * torch.randn_like(fake_for_disc)

        real_labels = torch.full((real_for_disc.size(0),), smooth_real, device=device)
        fake_labels = torch.zeros(fake_for_disc.size(0), device=device)

        # Train Discriminator
        disc_real = disc(real_for_disc)
        lossD_real = criterion(disc_real, real_labels)
        disc_fake = disc(fake_for_disc)
        lossD_fake = criterion(disc_fake, fake_labels)
        lossD = (lossD_real + lossD_fake) / 2

        disc.zero_grad()
        lossD.backward()
        disc_opt.step()

        # Train Generator
        output = disc(fake)
        lossG = criterion(output, real_labels)

        gen.zero_grad()
        lossG.backward()
        gen_opt.step()

    print(f"Epoch [{epoch+1}/{epochs}]  Loss D: {lossD:.4f}, loss G: {lossG:.4f}")

    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            varied_noise = torch.randn(64, latent_dim, 1, 1, device=device)
            fake = gen(varied_noise).detach().cpu()
            fake_grid = make_grid(fake, normalize=True)
            save_image(fake_grid, f"outputs/generated/fake_epoch_{epoch+1}.png")
            plt.figure(figsize=(8, 8))
            plt.imshow(fake_grid.permute(1, 2, 0))
            plt.title(f"Generated Samples at Epoch {epoch+1}")
            plt.axis("off")
            plt.show()

    
