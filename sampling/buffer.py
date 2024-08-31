import torch

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn, Tensor


@dataclass
class StoreCentroidsRequest:
    alive_centroids: List[Tensor] = None
    dead_centroids: List[Tensor] = None


@dataclass
class StoreQuerySamplesRequest:
    samples: List[Tensor] = None


class Buffer(torch.nn.Module):
    def __init__(self):
        super().__init__()
