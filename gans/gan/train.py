#%%
import torch
import os
from datetime import datetime

from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from gan import Generator, Discriminator

#%%
cfg = {
    # "lr": 0.0002,
    "lr": 1e-3,
    "bs": 128,
    "p": 0.1,
    "seed": 42,
    "epochs": 10,
    "k": 1,
    "checkpoint_interval": 1,  # Save samples every N epochs
    "n_samples": 100
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(cfg["seed"])
torch.set_float32_matmul_precision("high")

# %%
ROOT = os.path.join(os.path.dirname(__file__), "../../data")
dataset = MNIST(
    root=ROOT,
    train=True,
    download=True,
    transform=v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((32, 32)),
        v2.Normalize(mean=(0.5, ), std=(0.5, )),
        
    ])
)
dataloader = DataLoader(
    dataset,
    batch_size=cfg["bs"],
    shuffle=True,
)
# %%
# Plot first image from EMNIST dataset
plt.figure(figsize=(4,4))
img = dataset[2][0].squeeze().numpy()  # Get first image and remove channel dimension
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title("First EMNIST Image")
plt.show()
dataset[2][1]

# %%
generator = Generator()
discriminator = Discriminator(p=cfg["p"])

generator.to(device)
discriminator.to(device)
# %%

criterion = nn.BCEWithLogitsLoss()
g_optimizer = Adam(generator.parameters(), lr=cfg["lr"], betas=(0.5, 0.999))
d_optimizer = Adam(discriminator.parameters(), lr=cfg["lr"], betas=(0.5, 0.999))

# %%
g_losses = []
d_losses = []
total_losses = []

# %%
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../outputs/gan", timestamp)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create directories for generations
os.makedirs(os.path.join(OUTPUT_DIR, 'gens'), exist_ok=True)
gen_images = []  # Store paths for GIF creation

epoch_g_losses = []
epoch_d_losses = []

# %%
for i in range(cfg["epochs"]):
    epoch_g_loss = 0
    epoch_d_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {i + 1}/{cfg['epochs']}", leave=False)
    for batch_idx, (data, _) in enumerate(pbar):
        data = data.to(device)
        bs = data.shape[0]
        pos_labels = torch.ones(bs, device=device)
        neg_labels = torch.zeros(bs, device=device)
        
        # train discriminator
        d_loss = 0
        for _ in range(cfg["k"]):
            z = torch.randn(bs, 16, device=device)
            fake = generator(z)
            real_logits, fake_logits = discriminator(data), discriminator(fake.detach())
            disc_loss = criterion(
                real_logits,
                pos_labels
            ) + criterion(fake_logits, neg_labels)

            d_optimizer.zero_grad()
            disc_loss.backward()
            d_optimizer.step()
            
            d_loss += disc_loss
        
        d_loss = d_loss.mean()

        # train generator
        z = torch.randn(bs, 16, device=device)
        fake = generator(z)
        fake_logits = discriminator(fake)
        g_loss = criterion(fake_logits, pos_labels)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        loss = g_loss + d_loss
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        total_losses.append(loss.item())
        pbar.set_postfix(loss=loss.item(), g_loss=g_loss.item(), d_loss=d_loss.item())

        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()

    # Print epoch losses
    print(f"Epoch {i+1}/{cfg['epochs']} - Loss: {loss:.4f} G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}")

    # Generate samples at checkpoint intervals
    if (i + 1) % cfg["checkpoint_interval"] == 0:
        with torch.no_grad():
            z = torch.randn(cfg["n_samples"], 16, device=device)
            samples = generator(z)
            samples = samples.cpu()
            
            # Create a 10x10 grid
            fig, axs = plt.subplots(10, 10, figsize=(20, 20))
            plt.subplots_adjust(wspace=0, hspace=0)  # Remove spacing between subplots
            for idx, ax in enumerate(axs.flat):
                ax.imshow(samples[idx].permute(1, 2, 0).numpy(), cmap='gray')
                ax.axis('off')
            
            # Remove padding around the entire figure
            plt.tight_layout(pad=0)
            
            save_path = os.path.join(OUTPUT_DIR, 'gens', f'epoch_{i+1}.png')
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            gen_images.append(save_path)

# Save model weights
torch.save({
    'generator': generator.state_dict(),
    'discriminator': discriminator.state_dict(),
}, os.path.join(OUTPUT_DIR, 'checkpoint.pth'))

# %%
plt.figure(figsize=(10, 6))
plt.semilogy(g_losses, label='Generator Loss')
plt.semilogy(d_losses, label='Discriminator Loss')
plt.semilogy(total_losses, label='Total Loss')
plt.grid(True)
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Loss (log scale)')
plt.title('GAN Training Losses')
plt.savefig(os.path.join(OUTPUT_DIR, 'plots.png'))
plt.close()

# Create GIF from saved generations
images = []
for filename in gen_images:
    images.append(imageio.imread(filename))
imageio.mimsave(os.path.join(OUTPUT_DIR, 'gen.gif'), images, duration=3)

# %%
