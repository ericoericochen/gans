# gans

Implementation of Generative Adversarial Networks (GANs) and their variants in PyTorch.

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

$$V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)} \left[ \log D(x) \right] + \mathbb{E}_{z \sim p_z(z)} \left[ \log(1 - D(G(z))) \right]$$

### 2. Conditional GAN (CGAN)

- **Paper**: Conditional Generative Adversarial Nets (https://arxiv.org/abs/1411.1784)
- **Summary**: CGAN extends the standard GAN by conditioning both the Generator and Discriminator on additional information, such as class labels. This allows the Generator to produce data that is conditioned on specific attributes, enabling control over the generated output.

- **Objective Function**:

  The objective is similar to the GAN, but both the Generator and Discriminator are conditioned on the additional information y:

  V(D, G) = E[x, y ~ p_data(x, y)][log D(x|y)] + E[z, y ~ p_z(z), p_y(y)][log(1 - D(G(z|y), y))]

  Where y is the conditional label (e.g., class label).

### 3. Deep Convolutional GAN (DCGAN)

- **Paper**: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (https://arxiv.org/abs/1511.06434)
- **Summary**: DCGAN leverages deep convolutional architectures for both the Generator and Discriminator. This variant significantly improves the stability of GAN training, especially for image generation tasks.

- **Objective Function**:

  The objective function remains the same as the standard GAN, but with convolutional layers introduced to both networks to capture more complex spatial patterns in image data.

  V(D, G) = E[x ~ p_data(x)][log D(x)] + E[z ~ p_z(z)][log(1 - D(G(z)))]

  The difference lies in the architecture, not the objective function.

### 4. Wasserstein GAN (WGAN)

- **Paper**: Wasserstein GAN (https://arxiv.org/abs/1701.07875)
- **Summary**: WGAN introduces the Wasserstein distance as a metric to measure the divergence between the real and generated distributions. This method helps mitigate issues like mode collapse and provides more stable training.

- **Objective Function**:

  Instead of using the log loss from the original GAN, WGAN uses the Wasserstein distance between the real and generated data distributions:

  V(D, G) = E[x ~ p_data(x)][D(x)] - E[z ~ p_z(z)][D(G(z))]

  Where D(x) is the "critic" score, and the discriminator is replaced with a "critic" that does not output probabilities, but rather a scalar score. Training also includes weight clipping and potentially more critic updates.

### 5. Least Squares GAN (LSGAN)

- **Paper**: Least Squares Generative Adversarial Networks (https://arxiv.org/abs/1611.04076)
- **Summary**: LSGAN modifies the GAN objective by using least squares loss instead of cross-entropy loss. This approach helps stabilize training and produces higher-quality generated samples.

- **Objective Function**:

  The least squares loss is applied to both the real and generated data, aiming to minimize the least squares error:

  V(D, G) = E[x ~ p_data(x)][(D(x) - 1)^2] + E[z ~ p_z(z)][D(G(z))^2]

  This loss function penalizes the discriminator less severely than the original GAN's cross-entropy loss, helping stabilize training.

## Requirements

To run the code, you need to have the following dependencies:

- Python 3.7+
- PyTorch
- torchvision
- tqdm
- matplotlib
- imageio

## Installation

You can install the dependencies with uv.

## Usage

To train a specific model, you can run the training script with the desired configuration. Example:

```bash
python3 gans/gan.py --epochs 100 --lr 0.0002 --bs 64 --p 0.5 --k 5 --interval 10
```

For more information on each model and its specific configuration, refer to the respective files in the repository.
