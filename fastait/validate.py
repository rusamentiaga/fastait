import torch

def validate_images(images: torch.Tensor):
    """
    Validate that the input tensor is a stack of images with appropriate shape and type.

    Args:
        images (torch.Tensor): Input tensor to validate.

    Raises:
        ValueError: If the input tensor does not meet the required conditions.
    """
    if images.ndim != 3:
        raise ValueError("Input tensor must have shape (N, H, W)")
    if not images.is_floating_point():
        raise ValueError("Input tensor must be of float type")