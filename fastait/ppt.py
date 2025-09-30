import torch

def ppt(images: torch.Tensor) -> torch.Tensor:
    """
    Compute per-pixel FFT phase along the time dimension.

    Args:
        images (torch.Tensor): Input tensor (N, H, W)

    Returns:
        torch.Tensor: Phase spectrum (N, H, W), values in radians [-π, π]
    """
    # FFT along time axis (0), result: (N//2+1, H, W), complex-valued
    fft_out = torch.fft.rfft(images, dim=0)  

    # Compute phase (angle)
    phase = torch.angle(fft_out[1:])  # skip the DC bin
    
    return phase