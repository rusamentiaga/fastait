import torch
import numpy as np
from fastait.ppt import ppt

def test_ppt_vs_numpy():
    # Test parameters
    N, H, W = 8, 3, 3

    # Random test data
    np.random.seed(0)
    images = torch.randn(N, H, W, dtype=torch.float32)

    # Compute PyTorch phase
    phase_torch = ppt(images)

    # Compute NumPy phase
    images_np = images.numpy()
    fft_np = np.fft.rfft(images_np, axis=0)
    phase_np = np.angle(fft_np[1:])  # skip DC

    # Compare
    assert np.allclose(phase_torch.numpy(), phase_np, atol=1e-6)