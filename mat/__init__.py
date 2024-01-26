# From https://github.com/chaiNNer-org/spandrel
# This extends what ComfyUI copied out of chaiNNer's codebase.
# (until such time as it just uses spandrel as depndency)

from .arch.MAT import MAT


def load(state_dict):
    state = {
        k.replace("synthesis", "model.synthesis").replace("mapping", "model.mapping"): v
        for k, v in state_dict.items()
    }

    model = MAT()
    model.load_state_dict(state)
    return model
