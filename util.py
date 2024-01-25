from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor


def _gaussian_kernel(radius: int, sigma: float):
    x = torch.linspace(-radius, radius, steps=radius * 2 + 1)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    return pdf / pdf.sum()


def gaussian_blur(image: Tensor, radius: int, sigma: float = 0):
    c = image.shape[-3]
    if sigma <= 0:
        sigma = 0.3 * (radius - 1) + 0.8

    kernel = _gaussian_kernel(radius, sigma).to(image.device)
    kernel_x = kernel[..., None, :].repeat(c, 1, 1).unsqueeze(1)
    kernel_y = kernel[..., None].repeat(c, 1, 1).unsqueeze(1)

    image = F.pad(image, (radius, radius, radius, radius), mode="reflect")
    image = F.conv2d(image, kernel_x, groups=c)
    image = F.conv2d(image, kernel_y, groups=c)
    return image


def binary_erosion(mask: Tensor, radius: int):
    kernel = torch.ones(1, 1, radius * 2 + 1, radius * 2 + 1, device=mask.device)
    mask = F.pad(mask, (radius, radius, radius, radius), mode="constant", value=1)
    mask = F.conv2d(mask, kernel, groups=1)
    mask = (mask == kernel.numel()).to(mask.dtype)
    return mask


def make_odd(x):
    return x if x % 2 == 1 else x + 1
