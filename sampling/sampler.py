import torch
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn, Tensor


@dataclass
class SamplingRequest:
    target_ems: Tensor = None


@dataclass
class SamplingResult:
    # TrainEncoderRequest 와 동일
    positive_sample_embs: List[Tensor] = None
    positive_sample_scores: List[float] = None
    negative_sample_embs: List[Tensor] = None
    negative_sample_scores: List[float] = None


class Sampler(torch.nn.Module):
    def __init__(self):
        super().__init__()
