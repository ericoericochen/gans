import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, nc: int):
        super().__init__()
        self.nc = nc
        self.generator = nn.Sequential(
            nn.Linear(100 + nc, 1200),
            nn.ReLU(),
            nn.Linear(1200, 1200),
            nn.ReLU(),
            nn.Linear(1200, 784),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor):
        # generate onehot encoding for labels
        y = F.one_hot(y, num_classes=self.nc).to(device).float()
        x = torch.cat([z, y], dim=1)
        x = self.generator(x)
        x = x.sigmoid()

        return x


class Discriminator(nn.Module):
    def __init__(self, nc: int, p: float = 0.0):
        super().__init__()
        self.nc = nc
        self.discriminator = nn.Sequential(
            nn.Linear(784 + nc, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        y = F.one_hot(y, num_classes=self.nc).to(device).float()
        b = x.shape[0]
        x = x.view(b, -1)
        x = torch.cat([x, y], dim=1)
        logit = self.discriminator(x).view(-1)
        return logit


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--bs", type=int, default=128)
parser.add_argument("--p", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--k", type=int, default=1)
parser.add_argument("--interval", type=int, default=1)

try:
    args = parser.parse_args()  # Try to parse command line arguments
except:
    # If running in interactive mode (e.g. Jupyter), use defaults
    args = parser.parse_args([])

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
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)
dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True)

# Models
generator = Generator(nc=10).to(device)
discriminator = Discriminator(nc=10, p=args.p).to(device)

# Training setup
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = os.path.join(os.path.dirname(__file__), "../outputs/cgan", timestamp)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "gens"), exist_ok=True)

criterion = nn.BCEWithLogitsLoss()
g_optimizer = Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
d_optimizer = Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

# Training
g_losses, d_losses, total_losses = [], [], []
gen_images = []


for i in range(args.epochs):
    epoch_g_loss = epoch_d_loss = epoch_total_loss = n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {i + 1}/{args.epochs}", leave=False)
    for data, y in pbar:
        data = data.to(device)
        bs = data.shape[0]
        pos_labels = torch.ones(bs, device=device)
        neg_labels = torch.zeros(bs, device=device)

        # Train discriminator
        d_loss = 0
        for _ in range(args.k):
            z = torch.randn(bs, 100, device=device)
            fake = generator(z, y)
            real_logits, fake_logits = discriminator(data, y), discriminator(
                fake.detach(), y
            )
            disc_loss = criterion(real_logits, pos_labels) + criterion(
                fake_logits, neg_labels
            )

            d_optimizer.zero_grad()
            disc_loss.backward()
            d_optimizer.step()
            d_loss += disc_loss
        d_loss = d_loss.mean()

        # Train generator
        z = torch.randn(bs, 100, device=device)
        fake = generator(z, y)
        fake_logits = discriminator(fake, y)
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
            # Generate samples with labels 0-9 repeated across rows
            z = torch.randn(100, 100, device=device)
            y = torch.arange(10, device=device).repeat_interleave(10)
            samples = generator(z, y).cpu()

            fig, axs = plt.subplots(10, 10, figsize=(20, 20))
            plt.subplots_adjust(wspace=0, hspace=0)
            for idx, ax in enumerate(axs.flat):
                ax.imshow(samples[idx].view(28, 28).numpy(), cmap="gray")
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
imageio.mimsave(os.path.join(output_dir, "gen.gif"), images, duration=3)

print(f"Training complete. Outputs saved to: {output_dir}")
