import torch
from typing import List, Tuple, Any


def multiunsqueeze(t: torch.Tensor, dims: List[int]) -> torch.Tensor:
    """
    Recurseively unsqueeze the tensor along the dimensions in the list. Dimensions
    are unsqueezed from beginning to end of dims.
    """
    if len(dims) == 0:
        return t
    else:
        return multiunsqueeze(t.unsqueeze(dims[0]), dims[1:])

