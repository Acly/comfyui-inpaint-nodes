import folder_paths
import os

folder_paths.folder_names_and_paths.setdefault(
    "inpaint", ([os.path.join(folder_paths.models_dir, "inpaint")], [])
)[1].extend([".pt", ".pth", ".safetensors", ".patch"])

from . import nodes

NODE_CLASS_MAPPINGS = {
    "INPAINT_LoadFooocusInpaint": nodes.LoadFooocusInpaint,
    "INPAINT_ApplyFooocusInpaint": nodes.ApplyFooocusInpaint,
    "INPAINT_VAEEncodeInpaintConditioning": nodes.VAEEncodeInpaintConditioning,
    "INPAINT_MaskedBlur": nodes.MaskedBlur,
    "INPAINT_LoadInpaintModel": nodes.LoadInpaintModel,
    "INPAINT_InpaintWithModel": nodes.InpaintWithModel,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "INPAINT_LoadFooocusInpaint": "Load Fooocus Inpaint",
    "INPAINT_ApplyFooocusInpaint": "Apply Fooocus Inpaint",
    "INPAINT_VAEEncodeInpaintConditioning": "VAE Encode & Inpaint Conditioning",
    "INPAINT_MaskedBlur": "Blur Masked Area",
    "INPAINT_LoadInpaintModel": "Load Inpaint Model",
    "INPAINT_InpaintWithModel": "Inpaint (using Model)",
}
