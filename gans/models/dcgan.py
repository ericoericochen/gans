import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        channels: tuple[int] = (1024, 512, 258),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.channels = channels

        generator = []

        # project latent latent_dim x 1 x 1 to channels[0] x 4 x 4
        generator.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    latent_dim,
                    channels[0],
                    kernel_size=4,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(channels[0]),
                nn.ReLU(),
            )
        )

        for prev_ch, curr_ch in zip(channels[:-1], channels[1:]):
            generator.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        prev_ch, curr_ch, kernel_size=4, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(curr_ch),
                    nn.ReLU(),
                )
            )

        generator.append(
            nn.ConvTranspose2d(
                channels[-1], out_channels, kernel_size=4, stride=2, padding=1
            )
        )

        self.generator = nn.Sequential(*generator)

    def forward(self, z: torch.Tensor):
        x = self.generator(z)
        x = x.tanh()

        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels: int, channels: tuple[int] = (128, 256, 512, 1024)):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels

        discriminator = []
        discriminator.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    channels[0],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.LeakyReLU(0.2),
            )
        )

        for prev_ch, curr_ch in zip(channels[:-1], channels[1:]):
            discriminator.append(
                nn.Sequential(
                    nn.Conv2d(
                        prev_ch, curr_ch, kernel_size=3, stride=2, padding=1, bias=False
                    ),
                    nn.BatchNorm2d(curr_ch),
                    nn.LeakyReLU(0.2),
                )
            )

        discriminator.append(
            nn.Conv2d(channels[-1], 1, kernel_size=4, stride=1, padding=0, bias=False)
        )
        self.discriminator = nn.Sequential(*discriminator)

    def forward(self, x: torch.Tensor):
        b = x.shape[0]
        x = self.discriminator(x)
        x = x.view(b)

        return x


class DCGAN(nn.Module):
    @staticmethod
    def create(
        latent_dim: int,
        out_channels: int,
        generator_channels: tuple[int] = (1024, 512, 256, 128),
        discriminator_channels: tuple[int] = (128, 256, 512, 1024),
    ):
        assert (
            len(generator_channels) == len(discriminator_channels)
            and len(generator_channels) > 0
        )

        generator = Generator(latent_dim, out_channels, generator_channels)
        discriminator = Discriminator(out_channels, discriminator_channels)

        return DCGAN(generator, discriminator)

    def __init__(self, generator: Generator, discriminator: Discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.criterion = nn.BCEWithLogitsLoss()

        # initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0, 0.02)

    def generate(self, batch_size: int):
        device = next(self.parameters()).device
        z = torch.randn(
            (batch_size, self.generator.latent_dim, 1, 1),
            device=device,
        )
        x = self.generator(z)
        return x

    def discriminator_loss(self, x: torch.Tensor, data: torch.Tensor):
        b = x.shape[0]

        # create labels for real and fake data
        real_labels = torch.ones(b, device=x.device)
        fake_labels = torch.zeros(b, device=x.device)

        # predict logits
        real_logits = self.discriminator(data)
        fake_logits = self.discriminator(x)

        real_loss = self.criterion(real_logits, real_labels)
        fake_loss = self.criterion(fake_logits, fake_labels)

        loss = real_loss + fake_loss
        return loss

    def generator_loss(self, x: torch.Tensor):
        b = x.shape[0]
        labels = torch.ones(b, device=x.device)
        logits = self.discriminator(x)
        loss = self.criterion(logits, labels)

        return loss

    def forward(self, data: torch.Tensor):
        b = data.shape[0]
        x = self.generate(b)
        return x
