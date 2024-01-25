import folder_paths
import os

folder_paths.folder_names_and_paths.setdefault(
    "inpaint", ([os.path.join(folder_paths.models_dir, "inpaint")], [])
)[1].extend([".pth", ".safetensors", ".patch"])

from . import nodes

NODE_CLASS_MAPPINGS = {
    "INPAINT_LoadFooocusInpaint": nodes.LoadFooocusInpaint,
    "INPAINT_ApplyFooocusInpaint": nodes.ApplyFooocusInpaint,
    "INPAINT_FillInpaintArea": nodes.FillInpaintArea,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "INPAINT_LoadFooocusInpaint": "Load Fooocus Inpaint",
    "INPAINT_ApplyFooocusInpaint": "Apply Fooocus Inpaint",
    "INPAINT_FillInpaintArea": "Fill Inpaint Area (Blur)",
}
