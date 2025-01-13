# gans

Implementation of Generative Adversarial Networks (GANs) and their variants in PyTorch.

## Installation

You can install the dependencies with uv.

```
uv venv
source ./venv/bin/activate
uv pip install -r pyproject.toml
```

## Train Script

To train any of the GAN variants, you can run the following script:

```
python gans/[gan].py --epochs 10 --lr 0.001 --bs 128 --p 0.5 --k 1 --interval 1
```

### Arguments:

- --epochs: The number of training epochs (default: 10)
- --lr: The learning rate for the optimizer (default: 0.001)
- --bs: The batch size used during training (default: 128)
- --p: Probability for dropout or some other augmentation (depending on the model) (default: 0.5)
- --k: Number of discriminator updates per generator update ( default: 1)
- --interval: The number of iterations after which to log training progress (default: 1)

## Overview

### 1. GAN (Generative Adversarial Network)

- **Paper**: Generative Adversarial Nets (https://arxiv.org/abs/1406.2661)
- **Summary**: The standard GAN consists of two neural networks: a **Generator** and a **Discriminator**. The Generator tries to generate realistic data, while the Discriminator attempts to differentiate between real and fake data. Both networks are trained adversarially to improve each other.

$$V(D, G) = E_{x \sim p_{\text{data}}(x)} \left[ \log D(x) \right] + \mathbb{E}_{z \sim p_z(z)} \left[ \log(1 - D(G(z))) \right]$$

### 2. Conditional GAN (CGAN)

- **Paper**: Conditional Generative Adversarial Nets (https://arxiv.org/abs/1411.1784)
- **Summary**: CGAN extends the standard GAN by conditioning both the Generator and Discriminator on additional information, such as class labels. This allows the Generator to produce data that is conditioned on specific attributes, enabling control over the generated output.

- **Objective Function**:

  The objective is similar to the GAN, but both the Generator and Discriminator are conditioned on the additional information y:

  $$V(D, G) = E_{x, y \sim p_{\text{data}}(x, y)} \left[ \log D(x|y) \right] + \mathbb{E}_{z, y \sim p_z(z), p_y(y)} \left[ \log(1 - D(G(z|y))) \right]$$

  Where y is the conditional label (e.g., class label).

### 3. Deep Convolutional GAN (DCGAN)

- **Paper**: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (https://arxiv.org/abs/1511.06434)
- **Summary**: DCGAN uses deep convolutional architectures for both the Generator and Discriminator.

### 4. Wasserstein GAN (WGAN)

- **Paper**: Wasserstein GAN (https://arxiv.org/abs/1701.07875)
- **Summary**: WGAN introduces the Wasserstein distance as a metric to measure the divergence between the real and generated distributions. This method helps mitigate issues like mode collapse and provides more stable training.

![image](https://miro.medium.com/v2/resize:fit:1400/1*Yfa9bZL0d4NHaU1mHbGzjw.jpeg)

### 5. Least Squares GAN (LSGAN)

- **Paper**: Least Squares Generative Adversarial Networks (https://arxiv.org/abs/1611.04076)
- **Summary**: LSGAN modifies the GAN objective by using least squares loss instead of cross-entropy loss. This approach helps stabilize training and prevent vanishing gradients.

![image](https://miro.medium.com/v2/resize:fit:1400/1*nqOPfR5AV-1jLPmQ98WPLQ.png)

## Requirements

To run the code, you need to have the following dependencies:

- Python 3.7+
- PyTorch
- torchvision
- tqdm
- matplotlib
- imageio
