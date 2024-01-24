import folder_paths
import os

folder_paths.folder_names_and_paths["inpaint"] = (
    [os.path.join(folder_paths.models_dir, "inpaint")],
    [".pth", ".patch"],
)

from . import nodes

NODE_CLASS_MAPPINGS = {
    "INPAINT_LoadFooocusInpaint": nodes.LoadFooocusInpaint,
    "INPAINT_ApplyFooocusInpaint": nodes.ApplyFooocusInpaint,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "INPAINT_LoadFooocusInpaint": "Load Fooocus Inpaint",
    "INPAINT_ApplyFooocusInpaint": "Apply Fooocus Inpaint",
}
