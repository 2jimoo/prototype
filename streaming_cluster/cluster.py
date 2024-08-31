from river import cluster
from river import stream
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn, Tensor

# https://riverml.xyz/0.21.1/api/cluster/CluStream/


@dataclass
class AssignmentRequest:
    emb: Optional[Tensor] = None


class StreamCluster:
    def init(self):
        self.cluster = cluster.CluStream(
            n_macro_clusters=3, max_micro_clusters=5, time_gap=3, seed=0, halflife=0.4
        )

    def learn_and_assign(self, request: AssignmentRequest) -> int:
        self.clustream.learn_one(request.emb)
        return self.clustream.predict_one(request.emb)
