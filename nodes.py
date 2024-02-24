from __future__ import annotations
from typing import Any
import numpy as np
import torch
import torch.jit
import torch.nn.functional as F
from torch import Tensor
from tqdm import trange

from comfy.utils import ProgressBar
from comfy.model_patcher import ModelPatcher
from comfy.model_base import BaseModel
from comfy.model_management import cast_to_device, get_torch_device
import spandrel
from spandrel import nodes_upscale_model, ModelLoader
import comfy.utils
import comfy.lora
import folder_paths
import nodes

from . import mat
from .util import (
    gaussian_blur,
    binary_erosion,
    make_odd,
    to_torch,
    to_comfy,
    resize_square,
    undo_resize_square,
)


import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class InpaintHead(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize the convolution kernel parameter with a specific size
        self.head = nn.Parameter(torch.empty(320, 5, 3, 3))
        nn.init.kaiming_uniform_(self.head, a=math.sqrt(5))  # Use He initialization

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), "replicate")
        return F.conv2d(x, self.head)



def load_fooocus_patch(lora: dict, to_load: dict):
    patch_dict = {}
    loaded_keys = set()
    
    for key, value in to_load.items():
        if (patch := lora.get(value)) is not None:
            patch_dict[key] = ("fooocus", patch)
            loaded_keys.add(key)
    
    not_loaded = len([x for x in lora if x not in loaded_keys])
    print(f"[ApplyFooocusInpaint] {len(loaded_keys)} Lora keys loaded, {not_loaded} remaining keys not found in model.")
    
    return patch_dict



original_calculate_weight = ModelPatcher.calculate_weight
injected_model_patcher_calculate_weight = False

def inject_patched_calculate_weight():
    global injected_model_patcher_calculate_weight
    if not injected_model_patcher_calculate_weight:
        # Safely replacing the calculate_weight method with the patched version
        original_calculate_weight = ModelPatcher.calculate_weight
        
        def patched_calculate_weight(*args, **kwargs):
            return calculate_weight_patched(original_calculate_weight, *args, **kwargs)
        
        ModelPatcher.calculate_weight = patched_calculate_weight
        injected_model_patcher_calculate_weight = True
        print("[comfyui-inpaint-nodes] Patched calculate_weight method injected into ModelPatcher.")



def calculate_weight_patched(self, patches, weight, key):
    for patch in patches:
        alpha, v, _ = patch
        if isinstance(v, tuple) and v[0] == "fooocus":
            # Extracting the patch information
            _, patch_values = v
            w1 = torch.as_tensor(patch_values[0], device=weight.device, dtype=torch.float32)
            
            if w1.shape == weight.shape:
                # Applying the patch with alpha blending
                w_min = torch.as_tensor(patch_values[1], device=weight.device, dtype=torch.float32)
                w_max = torch.as_tensor(patch_values[2], device=weight.device, dtype=torch.float32)
                w1_normalized = (w1 / 255.0) * (w_max - w_min) + w_min
                weight.data += alpha * w1_normalized
            else:
                print(f"[ApplyFooocusInpaint] Shape mismatch {key}, weight not merged.")
        # If not a 'fooocus' patch or if any other conditions need to be checked, they can be added here
class LoadFooocusInpaint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "head": (folder_paths.get_filename_list("inpaint"),),
                "patch": (folder_paths.get_filename_list("inpaint"),),
            }
        }

    RETURN_TYPES = ("INPAINT_PATCH",)
    CATEGORY = "inpaint"
    FUNCTION = "load"

    def load(self, head: str, patch: str):
        head_file = folder_paths.get_full_path("inpaint", head)
        inpaint_head_model = InpaintHead()
        sd = torch.load(head_file, map_location="cpu")
        inpaint_head_model.load_state_dict(sd)

        patch_file = folder_paths.get_full_path("inpaint", patch)
        inpaint_lora = comfy.utils.load_torch_file(patch_file, safe_load=True)

        return (inpaint_head_model, inpaint_lora)

class ApplyFooocusInpaint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "patch": ("INPAINT_PATCH",),
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    CATEGORY = "inpaint"
    FUNCTION = "patch"

    def patch(self, model: ModelPatcher, patch: tuple[InpaintHead, dict[str, Tensor]], latent: dict[str, Any]):
        base_model: BaseModel = model.model
        latent_pixels = base_model.process_latent_in(latent["samples"])
        noise_mask = latent["noise_mask"].round()
        latent_mask = F.max_pool2d(noise_mask, (8, 8)).round().to(latent_pixels.device)

        inpaint_head_model, inpaint_lora = patch
        feed = torch.cat([latent_mask, latent_pixels], dim=1)
        inpaint_head_model.to(feed.device)
        inpaint_head_feature = inpaint_head_model(feed)

        def input_block_patch(h, transformer_options):
            if transformer_options["block"][1] == 0:
                h = h + inpaint_head_feature
            return h

        lora_keys = comfy.lora.model_lora_keys_unet(model.model, {})
        loaded_lora = load_fooocus_patch(inpaint_lora, lora_keys)

        m = model.clone()
        m.set_model_input_block_patch(input_block_patch)
        m.add_patches(loaded_lora, 1.0)

        not_patched_count = sum(1 for x in loaded_lora if x not in m)
        if not_patched_count > 0:
            print(f"[ApplyFooocusInpaint] Failed to patch {not_patched_count} keys")

        return m


class VAEEncodeInpaintConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "pixels": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "LATENT")
    FUNCTION = "encode"
    CATEGORY = "inpaint"

    def encode(self, positive, negative, vae, pixels, mask):
        # Assuming nodes.InpaintModelConditioning().encode(...) is correctly implemented elsewhere
        encoded = nodes.InpaintModelConditioning().encode(positive, negative, pixels, vae, mask)
        return encoded



class VAEEncodeInpaintConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "pixels": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "LATENT")
    FUNCTION = "encode"
    CATEGORY = "inpaint"

    def encode(self, positive, negative, vae, pixels, mask):
        # Assuming nodes.InpaintModelConditioning().encode(...) is correctly implemented elsewhere
        encoded = nodes.InpaintModelConditioning().encode(positive, negative, pixels, vae, mask)
        return encoded



class MaskedBlur:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "blur": ("INT", {"default": 255, "min": 3, "max": 8191, "step": 1}),
                "falloff": ("INT", {"default": 0, "min": 0, "max": 8191, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "inpaint"
    FUNCTION = "fill"

    def fill(self, image: Tensor, mask: Tensor, blur: int, falloff: int):
        blur = make_odd(blur)
        falloff = min(make_odd(falloff), blur - 2)

        image, mask = to_torch(image, mask)
        original = image.clone()
        alpha = mask.floor()

        if falloff > 0:
            erosion = binary_erosion(alpha, falloff)
            alpha = alpha * gaussian_blur(erosion, falloff)
        alpha = alpha.repeat(1, 3, 1, 1)

        image = gaussian_blur(image, blur)
        image = original + (image - original) * alpha

        return (to_comfy(image),)


class LoadInpaintModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("inpaint"),),
            }
        }

    RETURN_TYPES = ("INPAINT_MODEL",)
    CATEGORY = "inpaint"
    FUNCTION = "load"

    def load(self, model_name: str):
        model_file = folder_paths.get_full_path("inpaint", model_name)
        if model_file is None:
            raise RuntimeError(f"Model file not found: {model_name}")

        if model_file.endswith(".pt"):
            sd = torch.jit.load(model_file, map_location="cpu").state_dict()
        else:
            sd = comfy.utils.load_torch_file(model_file, safe_load=True)

        if "synthesis.first_stage.conv_first.conv.resample_filter" in sd:
            # MAT model
            model = mat.load(sd)
        else:
            # Spandrel model
            model = spandrel.load(sd)

        model = model.eval()
        return (model,)

class InpaintWithModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inpaint_model": ("INPAINT_MODEL",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "optional_upscale_model": ("UPSCALE_MODEL",),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "inpaint"
    FUNCTION = "inpaint"

    def inpaint(self, inpaint_model_path: str, image: Tensor, mask: Tensor, seed: int, optional_upscale_model_path=None):

        # Load inpainting model directly
        loader = ModelLoader()
        inpaint_model = loader.load_from_file(inpaint_model_path)

        # Upscaling setup (if necessary)
        if optional_upscale_model_path is not None:
            upscaler_model = loader.load_from_file(optional_upscale_model_path)
            upscaler = nodes_upscale_model  # Assuming you have 'nodes_upscale_model' defined elsewhere


        batch_size = image.shape[0]
        if mask.shape[0] != batch_size:
            mask = mask[0].unsqueeze(0).repeat(batch_size, 1, 1)

        batch_image = []
        pbar = ProgressBar(batch_size)
        for i in range(batch_size):
            work_image, work_mask = to_torch(image[i].unsqueeze(0), mask[i].unsqueeze(0))
            image_device = work_image.device
            original_image, original_mask = work_image, work_mask
            work_image, work_mask, original_size = resize_square(work_image, work_mask, required_size)
            work_mask = work_mask.floor()

            device = get_torch_device()
            inpaint_model.to(device)
            torch.manual_seed(seed)
            work_image = inpaint_model(work_image.to(device), work_mask.to(device))
            inpaint_model.cpu()

            if upscaler_model is not None:
                work_image = upscaler.upscale(upscaler, upscaler_model, work_image.movedim(1, -1))[0].movedim(-1, 1)
            work_image = undo_resize_square(work_image.to(image_device), original_size)
            work_image = original_image + (work_image - original_image) * original_mask.floor()
            batch_image.append(to_comfy(work_image))

            pbar.update(1)

        return (torch.cat(batch_image, dim=0),)