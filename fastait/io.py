import os
import torch
import numpy as np
from tqdm import tqdm
from natsort import natsorted  # natural sorting

def load_csv_folder(folder_path, dtype=torch.float32, use_cache=True, verbose=False):
    """
    Loads all CSV files in a folder representing images and returns a PyTorch tensor
    of shape (N, H, W), where:
        N = number of CSV files
        H, W = image height and width
    
    Args:
        folder_path (str): Path to the folder containing CSV files.
        dtype (torch.dtype): Desired dtype for the tensor.
        verbose (bool): If True, display a progress bar.
        cache_file (str or None): Path to save/load cached tensor. If exists, loads from cache.
    
    Returns:
        torch.Tensor: Tensor containing all images.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")
    
    if use_cache:
        folder_name = os.path.basename(os.path.normpath(folder_path))
        cache_file = os.path.join(folder_path, f"{folder_name}.pt")
    else:
        cache_file = None

    # Load from cache if it exists
    if cache_file and os.path.exists(cache_file):
        if verbose:
            print(f"Loading data from cache: {cache_file}")
        return torch.load(cache_file)    
    
    images = []
    
    # Sort CSV files naturally
    csv_files = natsorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])
    if not csv_files:
        raise ValueError(f"No CSV files found in '{folder_path}'")
    
    iterator = tqdm(csv_files, desc="Loading CSV images") if verbose else csv_files
    
    for file in iterator:
        csv_path = os.path.join(folder_path, file)
        img_array = np.loadtxt(csv_path, delimiter=',')
        img_tensor = torch.tensor(img_array, dtype=dtype)
        images.append(img_tensor)
    
    tensor_images = torch.stack(images)
    
    if cache_file:
        if verbose:
            print(f"Saving data to cache: {cache_file}")
        torch.save(tensor_images, cache_file)
    
    return tensor_images