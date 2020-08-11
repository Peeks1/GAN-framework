import numpy as np
import torch.cuda
import torch.autograd as autograd
import torch.nn as nn

from hyperparameters import image_size, latent_dim, fmsG, fmsD

cuda = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# inputs for ConvTranspose2d and Conv2d goes: input channels, output channels, kernel size, stride, padding


class Generator4x4(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input of function is vector of length latent_dim
            nn.ConvTranspose2d(latent_dim, image_size[0], 4, 2, 1, bias=False),
            nn.Tanh()
            # final state: image_size tensor
        )

    def forward(self, x):
        return self.main(x)


class Generator8x8(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input of function is vector of length latent_dim
            nn.ConvTranspose2d(latent_dim, fmsG, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmsG),
            nn.ReLU(True),
            # current state: fmsG x 4 x 4 tensor
            nn.ConvTranspose2d(fmsG, image_size[0], 4, 2, 1, bias=False),
            nn.Tanh()
            # final state: image_size tensor
        )

    def forward(self, x):
        return self.main(x)


class Generator16x16(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input of function is vector of length latent_dim
            nn.ConvTranspose2d(latent_dim, fmsG * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmsG * 2),
            nn.ReLU(True),
            # current state: (fmsG * 2) x 4 x 4 tensor
            nn.ConvTranspose2d(fmsG * 2, fmsG, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmsG),
            nn.ReLU(True),
            # current state: fmsG x 8 x 8 tensor
            nn.ConvTranspose2d(fmsG, image_size[0], 4, 2, 1, bias=False),
            nn.Tanh()
            # final state: image_size tensor
        )

    def forward(self, x):
        return self.main(x)


class Generator32x32(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input of function is vector of length latent_dim
            nn.ConvTranspose2d(latent_dim, fmsG * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmsG * 4),
            nn.ReLU(True),
            # current state: (fmsG * 4) x 4 x 4 tensor
            nn.ConvTranspose2d(fmsG * 4, fmsG * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmsG * 2),
            nn.ReLU(True),
            # current state: (fmsG * 2) x 8 x 8 tensor
            nn.ConvTranspose2d(fmsG * 2, fmsG, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmsG),
            nn.ReLU(True),
            # current state: fmsG x 16 x 16 tensor
            nn.ConvTranspose2d(fmsG, image_size[0], 4, 2, 1, bias=False),
            nn.Tanh()
            # final state: image_size tensor
        )

    def forward(self, x):
        return self.main(x)


class Generator64x64(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input of function is vector of length latent_dim
            nn.ConvTranspose2d(latent_dim, fmsG * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(fmsG * 8),
            nn.ReLU(True),
            # current state: (fmsG * 8) x 4 x 4 tensor
            nn.ConvTranspose2d(fmsG * 8, fmsG * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmsG * 4),
            nn.ReLU(True),
            # current state: (fmsG * 4) x 8 x 8 tensor
            nn.ConvTranspose2d(fmsG * 4, fmsG * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmsG * 2),
            nn.ReLU(True),
            # current state: (fmsG * 2) x 16 x 16 tensor
            nn.ConvTranspose2d(fmsG * 2, fmsG, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmsG),
            nn.ReLU(True),
            # current state: fmsG x 32 x 32 tensor
            nn.ConvTranspose2d(fmsG, image_size[0], 4, 2, 1, bias=False),
            nn.Tanh()
            # final state: image_size tensor
        )

    def forward(self, x):
        return self.main(x)


class Generator128x128(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input of function is vector of length latent_dim
            nn.ConvTranspose2d(latent_dim, fmsG * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(fmsG * 16),
            nn.ReLU(True),
            # current state: (fmsG * 16) x 4 x 4 tensor
            nn.ConvTranspose2d(fmsG * 16, fmsG * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmsG * 8),
            nn.ReLU(True),
            # current state: (fmsG * 8) x 4 x 4 tensor
            nn.ConvTranspose2d(fmsG * 8, fmsG * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmsG * 4),
            nn.ReLU(True),
            # current state: (fmsG * 4) x 8 x 8 tensor
            nn.ConvTranspose2d(fmsG * 4, fmsG * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmsG * 2),
            nn.ReLU(True),
            # current state: (fmsG * 2) x 16 x 16 tensor
            nn.ConvTranspose2d(fmsG * 2, fmsG, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmsG),
            nn.ReLU(True),
            # current state: fmsG x 32 x 32 tensor
            nn.ConvTranspose2d(fmsG, image_size[0], 4, 2, 1, bias=False),
            nn.Tanh()
            # final state: image_size tensor
        )

    def forward(self, x):
        return self.main(x)


class Discriminator4x4(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input is image_size tensor
            nn.Conv2d(image_size[0], 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # final state: 1 x 1 x 1 tensor
        )

    def forward(self, x):
        return self.main(x)


class Discriminator8x8(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input is image_size tensor
            nn.Conv2d(image_size[0], fmsD, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # current state: fmsD x 4 x 4 tensor
            nn.Conv2d(fmsD, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # final state: 1 x 1 x 1 tensor
        )

    def forward(self, x):
        return self.main(x)


class Discriminator16x16(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input is image_size tensor
            nn.Conv2d(image_size[0], fmsD, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # current state: fmsD x 8 x 8 tensor
            nn.Conv2d(fmsD, fmsD * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmsD * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # current state: (fmsD * 2) x 4 x 4 tensor
            nn.Conv2d(fmsD * 2, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # final state: 1 x 1 x 1 tensor
        )

    def forward(self, x):
        return self.main(x)


class Discriminator32x32(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input is image_size tensor
            nn.Conv2d(image_size[0], fmsD, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # current state: fmsD x 16 x 16 tensor
            nn.Conv2d(fmsD, fmsD * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmsD * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # current state: (fmsD * 2) x 8 x 8 tensor
            nn.Conv2d(fmsD * 2, fmsD * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmsD * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # current state: (fmsD * 4) x 4 x 4 tensor
            nn.Conv2d(fmsD * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # final state: 1 x 1 x 1 tensor
        )

    def forward(self, x):
        return self.main(x)


class Discriminator64x64(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input is image_size tensor
            nn.Conv2d(image_size[0], fmsD, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # current state: fmsD x 32 x 32 tensor
            nn.Conv2d(fmsD, fmsD * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmsD * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # current state: (fmsD * 2) x 16 x 16 tensor
            nn.Conv2d(fmsD * 2, fmsD * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmsD * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # current state: (fmsD * 4) x 8 x 8 tensor
            nn.Conv2d(fmsD * 4, fmsD * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmsD * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # current state: (fmsD * 8) x 4 x 4 tensor
            nn.Conv2d(fmsD * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # final state: 1 x 1 x 1 tensor
        )

    def forward(self, x):
        return self.main(x)


class Discriminator128x128(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input is image_size tensor
            nn.Conv2d(image_size[0], fmsD, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # current state: fmsD x 64 x 64 tensor
            nn.Conv2d(fmsD, fmsD * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmsD * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # current state: (fmsD * 2) x 32 x 32 tensor
            nn.Conv2d(fmsD * 2, fmsD * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmsD * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # current state: (fmsD * 4) x 16 x 16 tensor
            nn.Conv2d(fmsD * 4, fmsD * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmsD * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # current state: (fmsD * 8) x 8 x 8 tensor
            nn.Conv2d(fmsD * 8, fmsD * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmsD * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # current state: (fmsD * 16) x 4 x 4 tensor
            nn.Conv2d(fmsD * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # final state: 1 x 1 x 1 tensor
        )

    def forward(self, x):
        return self.main(x)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., probOfNoise=.5):
        self.std = std
        self.mean = mean
        self.probOfNoise = probOfNoise

    def __call__(self, tensor):
        if np.random.random() < self.probOfNoise:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates).view(-1)
    real_samples = real_samples.shape[0]
    fake = Tensor(real_samples).fill_(1.0)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty