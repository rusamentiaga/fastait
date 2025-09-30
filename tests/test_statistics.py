import torch
import numpy as np
from fastait.statistics import skewness, kurtosis

def test_skewness():
    N, H, W = 6, 4, 4
    sequence = torch.tensor([1, 2, 3, 4, 5, 20], dtype=torch.float64).view(N, 1, 1)
    images = sequence.repeat(1, H, W)
    skew_img = skewness(images)
    assert torch.allclose(skew_img, torch.full((H, W), 1.630542782855803, dtype=torch.float64))

def test_kurtosis():
    N, H, W = 6, 4, 4
    sequence = torch.tensor([1, 2, 3, 4, 5, 20], dtype=torch.float64).view(N, 1, 1)
    images = sequence.repeat(1, H, W)
    kurt_img = kurtosis(images)
    assert torch.allclose(kurt_img, torch.full((H, W), 3.920967318241520, dtype=torch.float64))