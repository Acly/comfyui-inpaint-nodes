from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor


def to_torch(image: Tensor, mask: Tensor | None = None):
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    image = image.permute(0, 3, 1, 2)  # BHWC -> BCHW
    if mask is not None:
        if len(mask.shape) < 4:
            mask = mask.reshape(1, 1, mask.shape[-2], mask.shape[-1])
    if image.shape[2:] != mask.shape[2:]:
        raise ValueError(
            f"Image and mask must be the same size. {image.shape[2:]} != {mask.shape[2:]}"
        )
    return image, mask


def to_comfy(image: Tensor):
    return image.permute(0, 2, 3, 1)  # BCHW -> BHWC


def resize_square(image: Tensor, mask: Tensor, size: int):
    _, _, h, w = image.shape
    pad_w, pad_h, prev_size = 0, 0, w
    if w == size and h == size:
        return image, mask, (pad_w, pad_h, prev_size)

    if w < h:
        pad_w = h - w
        prev_size = h
    elif h < w:
        pad_h = w - h
        prev_size = w
    image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    mask = F.pad(mask, (0, pad_w, 0, pad_h), mode="reflect")

    if image.shape[-1] != size:
        image = F.interpolate(image, size=size, mode="nearest-exact")
        mask = F.interpolate(mask, size=size, mode="nearest-exact")

    return image, mask, (pad_w, pad_h, prev_size)


def undo_resize_square(image: Tensor, original_size: tuple[int, int, int]):
    _, _, h, w = image.shape
    pad_w, pad_h, prev_size = original_size
    if prev_size != w or prev_size != h:
        image = F.interpolate(image, size=prev_size, mode="bilinear")
    return image[:, :, 0 : prev_size - pad_h, 0 : prev_size - pad_w]


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
    if x > 0 and x % 2 == 0:
        return x + 1
    return x
