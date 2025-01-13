import torch
import torch.nn as nn
import os
from datetime import datetime
import argparse

from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import imageio.v2 as imageio


class Generator(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()

        def block(in_channels: int, out_channels: int, kernel_size: int, **kwargs):
            layers = [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, **kwargs),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            return nn.Sequential(*layers)

        self.generator = nn.Sequential(
            block(latent_dim, 1024, 4, stride=1, bias=False),
            block(1024, 512, 4, stride=2, padding=1, bias=False),
            block(512, 256, 4, stride=2, padding=1, bias=False),
            block(256, 128, 4, stride=2, padding=1, bias=False),
            nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor):
        x = self.generator(z)
        x = x.tanh()
        return x


class Discriminator(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()

        def block(in_channels: int, out_channels: int, kernel_size: int, **kwargs):
            norm = kwargs.pop("norm", True)
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)]
            if norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout2d(p))
            return nn.Sequential(*layers)

        self.discriminator = nn.Sequential(
            block(1, 128, 3, stride=2, padding=1, norm=False),
            block(128, 256, 3, stride=2, padding=1, bias=False),
            block(256, 512, 3, stride=2, padding=1, bias=False),
            block(512, 1024, 3, stride=2, padding=1, bias=False),
            nn.Conv2d(1024, 1, 4, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor):
        logit = self.discriminator(x).view(-1)
        return logit


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--latent_dim", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--bs", type=int, default=128)
parser.add_argument("--p", type=float, default=0.3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--k", type=int, default=1)
parser.add_argument("--interval", type=int, default=1)
args = parser.parse_args()

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
torch.set_float32_matmul_precision("high")

# Data
ROOT = os.path.join(os.path.dirname(__file__), "../data")
dataset = MNIST(
    root=ROOT,
    train=True,
    download=True,
    transform=v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((64, 64)),
            v2.Normalize((0.5,), (0.5,)),
        ]
    ),
)
dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True)

# Models
generator = Generator(latent_dim=args.latent_dim).to(device)
discriminator = Discriminator(p=args.p).to(device)

# Training setup
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = os.path.join(os.path.dirname(__file__), "../outputs/lsgan", timestamp)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "gens"), exist_ok=True)

criterion = nn.MSELoss()
g_optimizer = Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
d_optimizer = Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

# Training
g_losses, d_losses, total_losses = [], [], []
gen_images = []


for i in range(args.epochs):
    epoch_g_loss = epoch_d_loss = epoch_total_loss = n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {i + 1}/{args.epochs}", leave=False)
    for data, _ in pbar:
        data = data.to(device)
        bs = data.shape[0]
        pos_labels = torch.ones(bs, device=device)
        neg_labels = torch.zeros(bs, device=device)

        # Train discriminator
        d_loss = 0
        for _ in range(args.k):
            z = torch.randn(bs, args.latent_dim, 1, 1, device=device)
            fake = generator(z)
            real_logits, fake_logits = discriminator(data), discriminator(fake.detach())
            disc_loss = criterion(real_logits, pos_labels) + criterion(
                fake_logits, neg_labels
            )

            d_optimizer.zero_grad()
            disc_loss.backward()
            d_optimizer.step()
            d_loss += disc_loss
        d_loss = d_loss.mean()

        # Train generator
        z = torch.randn(bs, args.latent_dim, 1, 1, device=device)
        fake = generator(z)
        real_logits = discriminator(data)
        fake_logits = discriminator(fake)
        g_loss = criterion(fake_logits, pos_labels)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Track losses
        loss = g_loss + d_loss
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        total_losses.append(loss.item())

        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()
        epoch_total_loss += loss.item()
        n_batches += 1

        pbar.set_postfix(loss=loss.item(), g_loss=g_loss.item(), d_loss=d_loss.item())

    # Print epoch stats
    epoch_g_loss /= n_batches
    epoch_d_loss /= n_batches
    epoch_total_loss /= n_batches
    print(
        f"Epoch {i+1}/{args.epochs} - Loss: {epoch_total_loss:.4f} G Loss: {epoch_g_loss:.4f}, D Loss: {epoch_d_loss:.4f}"
    )

    # Generate samples
    if (i + 1) % args.interval == 0:
        with torch.no_grad():
            z = torch.randn(100, args.latent_dim, 1, 1, device=device)
            samples = generator(z).cpu()

            fig, axs = plt.subplots(10, 10, figsize=(20, 20))
            plt.subplots_adjust(wspace=0, hspace=0)
            for idx, ax in enumerate(axs.flat):
                ax.imshow(samples[idx].permute(1, 2, 0).numpy(), cmap="gray")
                ax.axis("off")

            plt.tight_layout(pad=0)
            save_path = os.path.join(output_dir, "gens", f"epoch_{i+1}.png")
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
            plt.close()
            gen_images.append(save_path)

# Save final outputs
torch.save(
    {
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
    },
    os.path.join(output_dir, "checkpoint.pth"),
)

plt.figure(figsize=(10, 6))
plt.semilogy(g_losses, label="Generator Loss")
plt.semilogy(d_losses, label="Discriminator Loss")
plt.semilogy(total_losses, label="Total Loss")
plt.grid(True)
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Loss (log scale)")
plt.title("GAN Training Losses")
plt.savefig(os.path.join(output_dir, "plots.png"))
plt.close()

images = []
for filename in gen_images:
    images.append(imageio.imread(filename))

with imageio.get_writer(os.path.join(output_dir, "gen.gif"), duration=5) as writer:
    for image in images:
        writer.append_data(image)

print(f"Training complete. Outputs saved to: {output_dir}")
