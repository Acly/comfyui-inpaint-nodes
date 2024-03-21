import folder_paths
import os


def _add_folder_path(folder_name: str, extensions: list):
    path = os.path.join(folder_paths.models_dir, folder_name)
    _, current_extensions = folder_paths.folder_names_and_paths.setdefault(
        folder_name, ([path], set())
    )
    if isinstance(current_extensions, set):
        current_extensions.update(extensions)
    elif isinstance(current_extensions, list):
        current_extensions.extend(extensions)
    else:
        e = f"Failed to register models/inpaint folder. Found existing value: {current_extensions}"
        raise Exception(e)


_add_folder_path("inpaint", [".pt", ".pth", ".safetensors", ".patch"])

from . import nodes

NODE_CLASS_MAPPINGS = {
    "INPAINT_LoadFooocusInpaint": nodes.LoadFooocusInpaint,
    "INPAINT_ApplyFooocusInpaint": nodes.ApplyFooocusInpaint,
    "INPAINT_VAEEncodeInpaintConditioning": nodes.VAEEncodeInpaintConditioning,
    "INPAINT_MaskedFill": nodes.MaskedFill,
    "INPAINT_MaskedBlur": nodes.MaskedBlur,
    "INPAINT_LoadInpaintModel": nodes.LoadInpaintModel,
    "INPAINT_InpaintWithModel": nodes.InpaintWithModel,
    "INPAINT_DenoiseToCompositingMask": nodes.DenoiseToCompositingMask,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "INPAINT_LoadFooocusInpaint": "Load Fooocus Inpaint",
    "INPAINT_ApplyFooocusInpaint": "Apply Fooocus Inpaint",
    "INPAINT_VAEEncodeInpaintConditioning": "VAE Encode & Inpaint Conditioning",
    "INPAINT_MaskedFill": "Fill Masked Area",
    "INPAINT_MaskedBlur": "Blur Masked Area",
    "INPAINT_LoadInpaintModel": "Load Inpaint Model",
    "INPAINT_InpaintWithModel": "Inpaint (using Model)",
    "INPAINT_DenoiseToCompositingMask": "Denoise to Compositing Mask",
}
