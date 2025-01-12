# %%
%load_ext autoreload
%autoreload 2
# %%
import sys
sys.path.append("../")
import torch
import matplotlib.pyplot as plt


from gans.models.dcgan import Generator, Discriminator, DCGAN
from torchvision.transforms import v2
from medmnist import ChestMNIST

# %%
train_dataset = ChestMNIST(split='train', 
                          transform=v2.Compose([
                              v2.ToImage(),
                              v2.ToDtype(torch.float32, scale=True),
                              v2.Lambda(lambda x: 2 * x - 1)
                          ]),
                          size=64, 
                          download=True)
train_dataset
# %%
plt.imshow(train_dataset[0][0][0], cmap='gray')

# %%
train_dataset[0][0].min(), train_dataset[0][0].max()

# %%
generator = Generator(latent_dim=100, channels=(1024, 512, 256, 128), out_channels=1)
discriminator = Discriminator(in_channels=1, channels=(128, 256, 512, 1024))
discriminator
# %%
z = torch.randn(1, 100, 1, 1)
x = generator(z)
x.shape
# %%
logit = discriminator(x)
logit.shape

# %%
data = train_dataset[0][0].unsqueeze(0)
data.shape

# %%
dcgan = DCGAN.create(
    latent_dim=100,
    out_channels=1,
    generator_channels=(1024, 512, 256, 128),
    discriminator_channels=(128, 256, 512, 1024),
)

dcgan = dcgan.to("cuda")

dcgan

# %%
from tqdm import tqdm

gen_optimizer = torch.optim.Adam(
    dcgan.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
)
disc_optimizer = torch.optim.Adam(
    dcgan.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
)

pbar = tqdm(range(2000))
for i in pbar:
    batch = data.repeat(128, 1, 1, 1).to("cuda")

    x = dcgan(batch)
    disc_loss = dcgan.discriminator_loss(x.detach(), batch)
    disc_optimizer.zero_grad()
    disc_loss.backward()
    disc_optimizer.step()

    gen_loss = dcgan.generator_loss(x)
    loss = disc_loss + gen_loss

    gen_optimizer.zero_grad()
    gen_loss.backward()
    gen_optimizer.step()

    pbar.set_postfix(loss=loss.item(), disc_loss=disc_loss.item(), gen_loss=gen_loss.item())
# %%
x = dcgan.generate(1)
plt.imshow(x[0][0].detach().cpu().numpy(), cmap='gray')
# %%
dcgan.discriminator(x).sigmoid()
# %%
dcgan.discriminator(train_dataset[0][0].unsqueeze(0).to("cuda")).sigmoid()

# %%
real_data = train_dataset[0][0].unsqueeze(0).to("cuda")
generated_data = dcgan.generate(1).detach().cpu()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(real_data[0][0].cpu().numpy(), cmap='gray')
axes[0].set_title("Real Data")
axes[1].imshow(generated_data[0][0].numpy(), cmap='gray')
axes[1].set_title("Generated Data")
plt.show()

# %%
