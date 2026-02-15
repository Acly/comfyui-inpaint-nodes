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
from comfy import model_management
from comfy_api.latest import io
import comfy.utils
import comfy.lora
import folder_paths
import nodes

from . import mat
from .util import (
    BlurKernel,
    mask_blur,
    gaussian_blur,
    binary_erosion,
    binary_dilation,
    make_odd,
    mask_floor,
    mask_to_torch,
    to_torch,
    to_comfy,
    resize_square,
    undo_resize_square,
)


class InpaintHead(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = torch.nn.Parameter(torch.empty(size=(320, 5, 3, 3), device="cpu"))

    def __call__(self, x):
        x = F.pad(x, (1, 1, 1, 1), "replicate")
        return F.conv2d(x, weight=self.head)


def load_fooocus_patch(lora: dict, to_load: dict):
    patch_dict = {}
    loaded_keys = set()
    for key in to_load.values():
        if value := lora.get(key, None):
            patch_dict[key] = ("fooocus", value)
            loaded_keys.add(key)

    not_loaded = sum(1 for x in lora if x not in loaded_keys)
    if not_loaded > 0:
        print(
            f"[ApplyFooocusInpaint] {len(loaded_keys)} Lora keys loaded, {not_loaded} remaining keys not found in model."
        )
    return patch_dict


if not hasattr(comfy.lora, "calculate_weight") and hasattr(ModelPatcher, "calculate_weight"):
    too_old_msg = "comfyui-inpaint-nodes requires a newer version of ComfyUI (v0.1.1 or later), please update!"
    raise RuntimeError(too_old_msg)


original_calculate_weight = comfy.lora.calculate_weight
injected_model_patcher_calculate_weight = False


def calculate_weight_patched(
    patches, weight, key, intermediate_dtype=torch.float32, original_weights=None
):
    remaining = []

    for p in patches:
        alpha = p[0]
        v = p[1]

        is_fooocus_patch = isinstance(v, tuple) and len(v) == 2 and v[0] == "fooocus"
        if not is_fooocus_patch:
            remaining.append(p)
            continue

        if alpha != 0.0:
            v = v[1]
            w1 = cast_to_device(v[0], weight.device, torch.float32)
            if w1.shape == weight.shape:
                w_min = cast_to_device(v[1], weight.device, torch.float32)
                w_max = cast_to_device(v[2], weight.device, torch.float32)
                w1 = (w1 / 255.0) * (w_max - w_min) + w_min
                weight += alpha * cast_to_device(w1, weight.device, weight.dtype)
            else:
                print(
                    f"[ApplyFooocusInpaint] Shape mismatch {key}, weight not merged ({w1.shape} != {weight.shape})"
                )

    if len(remaining) > 0:
        return original_calculate_weight(remaining, weight, key, intermediate_dtype)
    return weight


def inject_patched_calculate_weight():
    global injected_model_patcher_calculate_weight
    if not injected_model_patcher_calculate_weight:
        print(
            "[comfyui-inpaint-nodes] Injecting patched comfy.model_patcher.ModelPatcher.calculate_weight"
        )
        comfy.lora.calculate_weight = calculate_weight_patched
        injected_model_patcher_calculate_weight = True


InpaintPatch = io.Custom("INPAINT_PATCH")


class LoadFooocusInpaint(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="INPAINT_LoadFooocusInpaint",
            display_name="Load Fooocus Inpaint",
            category="inpaint",
            inputs=[
                io.Combo.Input("head", folder_paths.get_filename_list("inpaint")),
                io.Combo.Input("patch", folder_paths.get_filename_list("inpaint")),
            ],
            outputs=[InpaintPatch.Output(display_name="inpaint patch")],
        )

    @classmethod
    def execute(cls, head: str, patch: str):  # type: ignore
        head_file = folder_paths.get_full_path("inpaint", head)
        assert head_file is not None, f"Inpaint head file not found in inpaint folder: {head}"
        inpaint_head_model = InpaintHead()
        sd = torch.load(head_file, map_location="cpu", weights_only=True)
        inpaint_head_model.load_state_dict(sd)

        patch_file = folder_paths.get_full_path("inpaint", patch)
        inpaint_lora = comfy.utils.load_torch_file(patch_file, safe_load=True)

        return io.NodeOutput(inpaint_head_model, inpaint_lora)


class InpaintBlockPatch:
    def __init__(self):
        self.inpaint_head_feature: Tensor | None = None
        self._inpaint_block: Tensor | None = None

    def __call__(self, h: Tensor, transformer_options: dict):
        if transformer_options["block"][1] == 0:
            if self._inpaint_block is None or self._inpaint_block.shape != h.shape:
                assert self.inpaint_head_feature is not None
                batch = h.shape[0] // self.inpaint_head_feature.shape[0]
                self._inpaint_block = self.inpaint_head_feature.to(h).repeat(batch, 1, 1, 1)
            h = h + self._inpaint_block
        return h


class ApplyFooocusInpaint(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="INPAINT_ApplyFooocusInpaint",
            display_name="Apply Fooocus Inpaint",
            category="inpaint",
            inputs=[
                io.Model.Input("model"),
                InpaintPatch.Input("patch", "inpaint patch"),
                io.Latent.Input("latent"),
            ],
            outputs=[io.Model.Output("model")],
        )

    @classmethod
    def execute(  # type: ignore
        cls,
        model: ModelPatcher,
        patch: tuple[InpaintHead, dict[str, Tensor]],
        latent: dict[str, Any],
    ):
        base_model: BaseModel = model.model
        latent_pixels = base_model.process_latent_in(latent["samples"])
        noise_mask = latent["noise_mask"].round()

        latent_mask = F.max_pool2d(noise_mask, (8, 8)).round().to(latent_pixels)

        inpaint_head_model, inpaint_lora = patch
        feed = torch.cat([latent_mask, latent_pixels], dim=1)
        inpaint_head_model.to(device=feed.device, dtype=feed.dtype)
        block_patch = InpaintBlockPatch()
        block_patch.inpaint_head_feature = inpaint_head_model(feed)

        lora_keys = comfy.lora.model_lora_keys_unet(model.model, {})
        lora_keys.update({x: x for x in base_model.state_dict().keys()})
        loaded_lora = load_fooocus_patch(inpaint_lora, lora_keys)

        m = model.clone()
        m.set_model_input_block_patch(block_patch)
        patched = m.add_patches(loaded_lora, 1.0)

        not_patched_count = sum(1 for x in loaded_lora if x not in patched)
        if not_patched_count > 0:
            print(f"[ApplyFooocusInpaint] Failed to patch {not_patched_count} keys")

        inject_patched_calculate_weight()
        return io.NodeOutput(m)


class VAEEncodeInpaintConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="INPAINT_VAEEncodeInpaintConditioning",
            display_name="VAE Encode & Inpaint Conditioning",
            category="inpaint",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Image.Input("pixels"),
                io.Mask.Input("mask"),
            ],
            outputs=[
                io.Conditioning.Output("positive"),
                io.Conditioning.Output("negative"),
                io.Latent.Output("latent_inpaint", "latent inpaint"),
                io.Latent.Output("latent_samples", "latent samples"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, pixels, mask):  # type: ignore
        try:
            positive, negative, latent = nodes.InpaintModelConditioning().encode(  # type: ignore
                positive, negative, pixels, vae, mask, noise_mask=True
            )
        except TypeError:  # ComfyUI versions older than 2024-11-19
            positive, negative, latent = nodes.InpaintModelConditioning().encode(  # type: ignore
                positive, negative, pixels, vae, mask
            )
        latent_inpaint = dict(
            samples=positive[0][1]["concat_latent_image"],
            noise_mask=latent["noise_mask"].round(),
        )
        return io.NodeOutput(positive, negative, latent_inpaint, latent)


class MaskedFill(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="INPAINT_MaskedFill",
            display_name="Fill Masked Area",
            category="inpaint",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Combo.Input("fill", ["neutral", "telea", "navier-stokes"]),
                io.Int.Input("falloff", default=0, min=0, max=8191, step=1),
            ],
            outputs=[io.Image.Output("image")],
        )

    @classmethod
    def execute(cls, image: Tensor, mask: Tensor, fill: str, falloff: int):  # type: ignore
        image = image.detach().clone()
        alpha = mask_to_torch(mask_floor(mask))
        alpha = binary_erosion(alpha, 10)
        assert alpha.shape[0] == image.shape[0], "Image and mask batch size does not match"

        falloff = make_odd(falloff)
        if falloff > 0:
            erosion = binary_erosion(alpha, falloff)
            alpha = alpha * gaussian_blur(erosion, falloff)

        if fill == "neutral":
            m = (1.0 - alpha).squeeze(1)
            for i in range(3):
                image[:, :, :, i] -= 0.5
                image[:, :, :, i] *= m
                image[:, :, :, i] += 0.5
        else:
            import cv2

            method = cv2.INPAINT_TELEA if fill == "telea" else cv2.INPAINT_NS
            for slice, alpha_slice in zip(image, alpha):
                alpha_np = alpha_slice.squeeze().cpu().numpy()
                alpha_bc = alpha_np.reshape(*alpha_np.shape, 1)
                image_np = slice.cpu().numpy()
                filled_np = cv2.inpaint(
                    (255.0 * image_np).astype(np.uint8),
                    (255.0 * alpha_np).astype(np.uint8),
                    3,
                    method,
                )
                filled_np = filled_np.astype(np.float32) / 255.0
                filled_np = image_np * (1.0 - alpha_bc) + filled_np * alpha_bc
                slice.copy_(torch.from_numpy(filled_np))

        return io.NodeOutput(image)


class MaskedBlur(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="INPAINT_MaskedBlur",
            display_name="Blur Masked Area",
            category="inpaint",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Int.Input("blur", default=255, min=3, max=8191, step=1),
                io.Int.Input("falloff", default=0, min=0, max=8191, step=1),
            ],
            outputs=[io.Image.Output("image")],
        )

    @classmethod
    def execute(cls, image: Tensor, mask: Tensor, blur: int, falloff: int):  # type: ignore
        blur = make_odd(blur)
        falloff = min(make_odd(falloff), blur - 2)
        image, mask = to_torch(image, mask)

        original = image.clone()
        alpha = mask_floor(mask)
        if falloff > 0:
            erosion = binary_erosion(alpha, falloff)
            alpha = alpha * gaussian_blur(erosion, falloff)
        alpha = alpha.expand(-1, 3, -1, -1)

        image = gaussian_blur(image, blur)
        image = original + (image - original) * alpha
        return io.NodeOutput(to_comfy(image))


InpaintModel = io.Custom("INPAINT_MODEL")


class LoadInpaintModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="INPAINT_LoadInpaintModel",
            display_name="Load Inpaint Model",
            category="inpaint",
            inputs=[
                io.Combo.Input("model_name", folder_paths.get_filename_list("inpaint")),
            ],
            outputs=[InpaintModel.Output(display_name="inpaint model")],
        )

    @classmethod
    def execute(cls, model_name: str):  # type: ignore
        from spandrel import ModelLoader

        model_file = folder_paths.get_full_path("inpaint", model_name)
        if model_file is None:
            raise RuntimeError(f"Model file not found: {model_name}")
        if model_file.endswith(".pt"):
            sd = torch.jit.load(model_file, map_location="cpu").state_dict()
        else:
            sd = comfy.utils.load_torch_file(model_file, safe_load=True)

        if "synthesis.first_stage.conv_first.conv.resample_filter" in sd:  # MAT
            model = mat.load(sd)
        else:
            model = ModelLoader().load_from_state_dict(sd)  # type: ignore
        model = model.eval()
        return io.NodeOutput(model)


class InpaintWithModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="INPAINT_InpaintWithModel",
            display_name="Inpaint (using Model)",
            category="inpaint",
            inputs=[
                InpaintModel.Input("inpaint_model"),
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, step=1),
                io.Model.Input("optional_upscale_model", "upscale model (optional)", optional=True),
            ],
            outputs=[io.Image.Output("inpainted image")],
        )

    @classmethod
    def execute(  # type: ignore
        cls,
        inpaint_model: mat.MAT | Any,
        image: Tensor,
        mask: Tensor,
        seed: int,
        optional_upscale_model=None,
    ):
        if isinstance(inpaint_model, mat.MAT):
            required_size = 512
        elif inpaint_model.architecture.id == "LaMa":
            required_size = 256
        else:
            raise ValueError(f"Unknown model_arch {type(inpaint_model)}")

        image, mask = to_torch(image, mask)
        batch_size = image.shape[0]
        if mask.shape[0] != batch_size:
            mask = mask[0].unsqueeze(0).repeat(batch_size, 1, 1, 1)

        image_device = image.device
        device = get_torch_device()
        inpaint_model.to(device)
        batch_image = []
        pbar = ProgressBar(batch_size)

        for i in trange(batch_size):
            work_image, work_mask = image[i].unsqueeze(0), mask[i].unsqueeze(0)
            work_image, work_mask, original_size = resize_square(
                work_image, work_mask, required_size
            )
            work_mask = mask_floor(work_mask)

            torch.manual_seed(seed)
            work_image = inpaint_model(work_image.to(device), work_mask.to(device))

            if optional_upscale_model is not None:
                work_image = cls._upscale(optional_upscale_model, work_image, device)

            work_image.to(image_device)
            work_image = undo_resize_square(work_image.to(image_device), original_size)
            work_image = image[i] + (work_image - image[i]) * mask_floor(mask[i])

            batch_image.append(work_image)
            pbar.update(1)

        inpaint_model.cpu()
        result = torch.cat(batch_image, dim=0)
        return io.NodeOutput(to_comfy(result))

    @classmethod
    def _upscale(cls, upscale_model, image: Tensor, device):
        memory_required = model_management.module_size(upscale_model.model)
        memory_required += (
            (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0
        )
        memory_required += image.nelement() * image.element_size()
        model_management.free_memory(memory_required, device)
        upscale_model.to(device)

        tile = 512
        overlap = 32
        oom = True
        s: Tensor | None = None
        while oom:
            try:
                s = comfy.utils.tiled_scale(
                    image,
                    lambda a: upscale_model(a),
                    tile_x=tile,
                    tile_y=tile,
                    overlap=overlap,
                    upscale_amount=upscale_model.scale,
                )
                oom = False
            except model_management.OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise e

        upscale_model.to("cpu")
        assert s is not None
        return torch.clamp(s, min=0, max=1.0)


class DenoiseToCompositingMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="INPAINT_DenoiseToCompositingMask",
            display_name="Denoise to Compositing Mask",
            category="inpaint",
            inputs=[
                io.Mask.Input("mask"),
                io.Float.Input("offset", default=0.1, min=0.0, max=1.0, step=0.01),
                io.Float.Input("threshold", default=0.2, min=0.01, max=1.0, step=0.01),
            ],
            outputs=[io.Mask.Output("mask")],
        )

    @classmethod
    def execute(cls, mask: Tensor, offset: float, threshold: float):  # type: ignore
        assert 0.0 <= offset < threshold <= 1.0, "Threshold must be higher than offset"
        mask = (mask - offset) * (1 / (threshold - offset))
        mask = mask.clamp(0, 1)
        return io.NodeOutput(mask)


class ExpandMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="INPAINT_ExpandMask",
            display_name="Expand Mask",
            category="inpaint",
            inputs=[
                io.Mask.Input("mask"),
                io.Int.Input("grow", default=16, min=0, max=8096, step=1),
                io.Int.Input("blur", default=7, min=0, max=8096, step=1),
                io.Combo.Input("blur_type", BlurKernel, default=BlurKernel.gaussian),
            ],
            outputs=[io.Mask.Output("mask")],
        )

    @classmethod
    def execute(cls, mask: Tensor, grow: int, blur: int, blur_type: BlurKernel | str):  # type: ignore
        mask = mask_to_torch(mask)
        if grow > 0:
            mask = binary_dilation(mask, grow)
        if blur > 0:
            blur_type = BlurKernel[blur_type] if isinstance(blur_type, str) else blur_type
            mask = mask_blur(mask, make_odd(blur), blur_type)
        return io.NodeOutput(mask.squeeze(1))


class ShrinkMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="INPAINT_ShrinkMask",
            display_name="Shrink Mask",
            category="inpaint",
            inputs=[
                io.Mask.Input("mask"),
                io.Int.Input("shrink", default=16, min=0, max=8096, step=1),
                io.Int.Input("blur", default=7, min=0, max=8096, step=1),
                io.Combo.Input("blur_type", BlurKernel, default=BlurKernel.gaussian),
            ],
            outputs=[io.Mask.Output("mask")],
        )

    @classmethod
    def execute(cls, mask: Tensor, shrink: int, blur: int, blur_type: BlurKernel | str):  # type: ignore
        mask = mask_to_torch(mask)
        if shrink > 0:
            mask = binary_erosion(mask, shrink)
        if blur > 0:
            blur_type = BlurKernel[blur_type] if isinstance(blur_type, str) else blur_type
            mask = mask_blur(mask, make_odd(blur), blur_type)
        return io.NodeOutput(mask.squeeze(1))


class StabilizeMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="INPAINT_StabilizeMask",
            display_name="Stabilize Mask",
            category="inpaint",
            inputs=[
                io.Mask.Input("mask"),
                io.Float.Input("epsilon", default=0.01, min=0.0, max=1.0, step=0.0001),
            ],
            outputs=[io.Mask.Output("mask")],
        )

    @classmethod
    def execute(cls, mask: Tensor, epsilon: float):  # type: ignore
        mask = mask_to_torch(mask)
        mask = torch.where(mask > 1.0 - epsilon, torch.ones_like(mask), mask)
        return io.NodeOutput(mask.squeeze(1))
