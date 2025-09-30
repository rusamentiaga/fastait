import torch
import fastait.validate

def normalize_percentile(images: torch.Tensor, percentile_low: float, percentile_high: float) -> torch.Tensor:
    """
    Normalizes each image in a tensor of shape (N, H, W) to the range [0, 1] based on the
    specified lower and upper percentiles.
    
    Args:
        images (torch.Tensor): Input tensor of shape (N, H, W).
        percentile_low (float): Lower percentile (0-100) for normalization.
        percentile_high (float): Upper percentile (0-100) for normalization.
    
    Returns:
        torch.Tensor: Normalized tensor of the same shape as input, clipped to [0, 1].
    """
    # Validate input tensor
    fastait.validate.validate_images(images)

    N, H, W = images.shape
    flat_images = images.view(N, -1)

    # Compute percentiles per image
    low_values = torch.quantile(flat_images, percentile_low / 100.0, dim=1, keepdim=True)
    high_values = torch.quantile(flat_images, percentile_high / 100.0, dim=1, keepdim=True)

    # Normalize to [0, 1]
    eps = torch.finfo(images.dtype).eps
    normalized = (flat_images - low_values) / (high_values - low_values + eps)
    normalized = torch.clamp(normalized, 0.0, 1.0)

    # Reshape back to original shape
    return normalized.view(N, H, W)

def extract_cooling(images: torch.Tensor) -> torch.Tensor:
    """
    Extract cooling events from a stack of images by thresholding the mean pixel value.

    Args:
        images (torch.Tensor): Tensor of shape (N, H, W)
        """
    fastait.validate.validate_images(images)

    mean_per_image = images.mean(dim=(1, 2))
    max_index = mean_per_image.argmax().item()
    if max_index + 1 >= images.shape[0]:
        raise ValueError("No cooling data found after the peak mean pixel value.")

    return images[max_index+1:]