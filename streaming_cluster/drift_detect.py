from river import drift
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn, Tensor

@dataclass
class DetectionRequest:
    something: Optional[Tensor] = None

class DriftDetector:
    def init(self):
        self.ph = drift.PageHinkley()

    def update_and_detect(self, request: DetectionRequest) -> bool:
        self.ph.update(request.something)
        return self.ph.drift_detected

# detection value를 얻을 수 있게 수정 
class PageHinkley:
    def __init__(self, sigma, lambda_):
        """
            :param sigma: 허용 노이즈 수준
            :param lambda_: 변화 감지 임계값
        """
        self.sigma = sigma
        self.lambda_ = lambda_
        self.y_sum = 0.0
        self.y_squared_sum = 0.0
        self.T = 0
        self.UT = 0.0
        self.mT = 0.0

    def update(self, y_t):
        """
            새로운 데이터 포인트 y_t가 들어올 때마다 호출됨
            :param y_t: 새로 들어온 데이터 포인트
        """
        self.T += 1
        self.y_sum += y_t
        self.y_squared_sum += y_t ** 2
        yT = self.y_sum / self.T
    
        self.UT += (y_t - yT - self.sigma)
        self.mT = min(self.mT, self.UT) if self.T > 1 else self.UT
        PHT = self.UT - self.mT

        return PHT 
