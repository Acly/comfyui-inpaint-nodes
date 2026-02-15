import folder_paths
import os
from comfy_api.latest import ComfyExtension, io


def _add_folder_path(folder_name: str, extensions_to_register: list):
    path = os.path.join(folder_paths.models_dir, folder_name)
    folders, extensions = folder_paths.folder_names_and_paths.get(folder_name, ([], set()))
    if path not in folders:
        folders.append(path)
    if isinstance(extensions, set):
        extensions.update(extensions_to_register)
    elif isinstance(extensions, list):
        extensions.extend(extensions_to_register)
    else:
        e = f"Failed to register models/inpaint folder. Found existing value: {extensions}"
        raise Exception(e)
    folder_paths.folder_names_and_paths[folder_name] = (folders, extensions)


_add_folder_path("inpaint", [".pt", ".pth", ".safetensors", ".patch"])

from . import nodes


class InpaintNodes(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            nodes.LoadFooocusInpaint,
            nodes.ApplyFooocusInpaint,
            nodes.VAEEncodeInpaintConditioning,
            nodes.MaskedFill,
            nodes.MaskedBlur,
            nodes.LoadInpaintModel,
            nodes.InpaintWithModel,
            nodes.ColorMatch,
            nodes.ExpandMask,
            nodes.ShrinkMask,
            nodes.StabilizeMask,
            nodes.DenoiseToCompositingMask,
        ]


async def comfy_entrypoint():
    return InpaintNodes()
