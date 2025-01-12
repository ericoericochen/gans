import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, c: int = 1):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1024 * c),
        )

    def forward(self, z: torch.Tensor):
        b = z.shape[0]
        x = self.generator(z)
        x = x.view(b, -1, 32, 32)
        x = x.tanh()

        return x


class Discriminator(nn.Module):
    def __init__(self, c: int = 1, p: float = 0.0):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(1024 * c, 256),  # 32*32 = 1024 input size
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True), 
            nn.Dropout(p),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor):
        b = x.shape[0]
        x = x.view(b, -1)
        logit = self.discriminator(x).view(-1)
        return logit
