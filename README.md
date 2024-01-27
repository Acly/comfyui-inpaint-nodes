# ComfyUI Inpaint Nodes

Nodes for better inpainting with ComfyUI: Fooocus inpaint model for SDXL, LaMa, MAT, and various other tools for pre-filling inpaint & outpaint areas.

## Fooocus Inpaint

Adds two nodes which allow using [Fooocus](https://github.com/lllyasviel/Fooocus) inpaint model. It's a small and flexible patch which can be applied to _any **SDXL** checkpoint_ and will transform it into an inpaint model. This model can then be used like other inpaint models, and provides the same benefits. [Read more](https://github.com/lllyasviel/Fooocus/discussions/414)

Download models from [lllyasviel/fooocus_inpaint](https://huggingface.co/lllyasviel/fooocus_inpaint/tree/main) to `ComfyUI/models/inpaint`.

![Inpaint workflow](media/inpaint.png)

Note: Implementation is somewhat hacky as it monkey-patches ComfyUI's `ModelPatcher` to support the custom Lora format which the model is using.

## Inpaint Conditioning

Fooocus inpaint can be used with ComfyUI's _VAE Encode (for Inpainting)_ directly. However this does not allow using existing content in the masked area, denoise strength must be 1.0.

Alternatively _InpaintModelConditioning_ can be used to allow using inpaint models with existing content. The resulting latent can however _not_ be used to patch the model using _Apply Fooocus Inpaint_. This repository provides a new node **VAE Encode & Inpaint Conditioning** which provides two outputs: `latent_inpaint` (connect this to _Apply Fooocus Inpaint_) and `latent_samples` (connect this to _KSampler_).

It's the same as using both _VAE Encode (for Inpainting)_ and _InpaintModelConditioning_, but less overhead because it avoids VAE-encoding the image twice.



## Inpaint Pre-processing

Several nodes are available to fill the masked area prior to inpainting.

### Fill Masked

This fills the masked area, with a smooth transition at the border. It has 3 modes:
* `neutral`: fills with grey, good for adding entirely new content
* `telea`: fills with colors from surrounding border (based on algorithm by Alexandru Telea)
* `navier-stokes`: fills with colors from surrounding border (based on fluid dynamics described by Navier-Stokes)

| Input | Neutral | Telea | Navier-Stokes |
|-|-|-|-|
| ![input](media/preprocess-input.png) | ![neutral](media/preprocess-neutral.png) | ![telea](media/preprocess-telea.png) | ![ns](media/preprocess-navier-stokes.png)

### Blur Masked

This blurs the image into the masked area. The blur is less strong at the borders of the mask. Good for keeping the general colors the same, but generating structurally new content.

| Input | Blur radius 17 | Blur radius 65 |
|-|-|-|
| ![input](media/preprocess-input.png) | ![blur-17](media/preprocess-blur-17.png) | ![blur-65](media/preprocess-blur-65.png) |

### Inpaint Models (LaMA, MAT)

This runs a small, fast inpaint model on the masked area. Models can be loaded with **Load Inpaint Model** and are applied with the **Inpaint (using Model)** node. This works well for outpainting or object removal.

The following inpaint models are supported, place them in `ComfyUI/models/inpaint`:
- [LaMa](https://github.com/advimman/lama) | [Model download](https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt)
- [MAT](https://github.com/fenglinglwb/MAT) | [Model download](https://github.com/Sanster/models/releases/download/add_mat/Places_512_FullData_G.pth)

| Input | LaMa | MAT |
|-|-|-|
| ![input](media/preprocess-input.png) | ![lama](media/preprocess-lama.png) | ![mat](media/preprocess-mat.png) |

## Example Workflows

Example workflows can be found in [workflows](workflows).

* **[Simple](https://raw.githubusercontent.com/Acly/comfyui-inpaint-nodes/main/workflows/inpaint-simple.json):** basic workflow, ignore previous content, 100% replacement
* **[Advanced](https://raw.githubusercontent.com/Acly/comfyui-inpaint-nodes/main/workflows/refine-advanced.json):** complex workflow, inpaint/outpaint/refine, 1-100% denoise strength


## Installation

Use [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) and search for "ComfyUI Inpaint Nodes".

_**or**_ download the repository and put the folder into `ComfyUI/custom_nodes`.

_**or**_ use GIT:
```
cd ComfyUI/custom_nodes
git clone https://github.com/Acly/comfyui-inpaint-nodes.git
```

Restart ComfyUI after installing!

---

OpenCV is required for _telea_ and _navier-stokes_ fill mode:
```
pip install opencv-python
```

## Acknowledgements

* Fooocus Inpaint: [lllyasviel/Fooocus](https://github.com/lllyasviel/Fooocus)
* LaMa: [advimman/lama](https://github.com/advimman/lama)
* MAT: [fenglinglwb/MAT](https://github.com/fenglinglwb/MAT)
* LaMa/MAT implementation: [chaiNNer-org/spandrel](https://github.com/chaiNNer-org/spandrel)
