import torch
import fastait.validate

def highorder_statistics(images: torch.Tensor, order):
    """
    Compute per-pixel high-order statistics (skewness for order=3, kurtosis for order=4)
    across a stack of images using PyTorch.
    Args:
        images (torch.Tensor): Tensor of shape (N, H, W)
        order (int): Order of the statistic (3 for skewness, 4 for kurtosis)
    Returns:
        stat (torch.Tensor): Image of the computed statistic (H, W)
    """
    fastait.validate.validate_images(images)
    
    # Compute mean per pixel
    mean = images.mean(dim=0)  # (H, W)
    
    # Compute standard deviation per pixel
    std = images.std(dim=0, unbiased=True)  # (H, W)
    
    # Compute high-order statistic
    stat = ((images - mean) / std).pow(order).mean(dim=0)
    
    return stat

def skewness(images: torch.Tensor):
    """
    Compute per-pixel skewness across a stack of images using PyTorch.
    Args:
        images (torch.Tensor): Tensor of shape (N, H, W)
    Returns:
        skewness (torch.Tensor): Image of skewness (H, W)
    """
    return highorder_statistics(images, order=3)

def kurtosis(images: torch.Tensor):
    """
    Compute per-pixel kurtosis across a stack of images using PyTorch.
    Args:
        images (torch.Tensor): Tensor of shape (N, H, W)
    Returns:
        kurtosis (torch.Tensor): Image of kurtosis (H, W)
    """
    return highorder_statistics(images, order=4)