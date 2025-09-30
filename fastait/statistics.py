import torch
import fastait.validate

def skewness(images: torch.Tensor):
    """
    Compute per-pixel skewness across a stack of images using PyTorch,
    compatible with MATLAB default skewness.

    Args:
        images (torch.Tensor): Tensor of shape (N, H, W)
    
    Returns:
        torch.Tensor: Skewness image of shape (H, W)
    """
    fastait.validate.validate_images(images)

    N = images.shape[0]
    mean = images.mean(dim=0)
    x0 = images - mean
    
    # Biased second and third moments
    s2 = (x0 ** 2).mean(dim=0)  # biased variance
    m3 = (x0 ** 3).mean(dim=0)  # third central moment
    
    skew = m3 / (s2 ** 1.5)
    return skew

def kurtosis(images: torch.Tensor):
    """
    Compute per-pixel kurtosis across a stack of images using PyTorch.
    compatible with MATLAB default kurtosis.

    Args:
        images (torch.Tensor): Tensor of shape (N, H, W)
    Returns:
        kurtosis (torch.Tensor): Image of kurtosis (H, W)
    """
    fastait.validate.validate_images(images)
    
    mean = images.mean(dim=0)
    x0 = images - mean
    s2 = torch.mean(x0**2, dim=0)      # biased variance
    m4 = torch.mean(x0**4, dim=0)
    k = m4 / (s2**2)
    return k