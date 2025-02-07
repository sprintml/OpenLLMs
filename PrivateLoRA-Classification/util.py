from typing import Any, Dict, Union
import torch
import random
import numpy as np
import os

def prepare_inputs(inputs: Dict[str, Union[torch.Tensor, Any]], device) -> Dict[str, Union[torch.Tensor, Any]]:
    """
    Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state.
    """
    if isinstance(inputs, list) :
        return None
    else :
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        return inputs
    
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
