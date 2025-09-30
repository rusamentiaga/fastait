import torch
import fastait.validate

def pct(images, n_components=None):
    """
    Perform Principal Component Thermography (PCT) on a stack of images using SVD.
    Args:
        data (torch.Tensor): Input tensor of shape (N, H, W)
        n_components (int, optional): Number of principal components to retain. 
            If None, all components are retained.
    Returns:
        torch.Tensor: Tensor of shape (q, H, W) where q is the number of components returned.
    """
    fastait.validate.validate_images(images)

    N, H, W = images.shape        # (time/frames, height, width)
    M = H * W                     # flattened spatial size

    # Flatten spatial dimensions -> shape (M, N)
    pixel_time = images.reshape(N, M).T  

    # Normalize along time
    mean = pixel_time.mean(dim=0)
    std = pixel_time.std(dim=0)
    eps = torch.finfo(pixel_time.dtype).eps
    pixel_time_normalized = (pixel_time - mean) / (std + eps)

    if n_components is None:
        # Full economy SVD
        U, _, _ = torch.linalg.svd(pixel_time_normalized, full_matrices=False)
    else:
        # Ensure q doesn't exceed the rank limit
        q = min(n_components, min(M, N))
        U, _, _ = torch.svd_lowrank(pixel_time_normalized, q=q)

    # Number of components actually returned
    output_components = U.shape[1]

    # Reshape to (output_components, H, W)
    data_pct = U.T.reshape(output_components, H, W)

    return data_pct

