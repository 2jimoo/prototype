from sklearn.cluster import BisectingKMeans
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn, Tensor

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.BisectingKMeans.html
# https://scikit-learn.org/stable/modules/clustering.html#bisecting-k-means


# 필드값 재정의
@dataclass
class EstimationRequest:
    isAlreted: bool
    isStationary: bool


class KEstimatior:
    def __init__(self):
        # 생구현 https://medium.com/@afrizalfir/bisecting-kmeans-clustering-5bc17603b8a2
        self.bisect_means = BisectingKMeans(n_clusters=3, random_state=0)

    def estimate_centroids(self, data):
        self.bisect_means.fit(data)
        return self.bisect_means.cluster_centers_
