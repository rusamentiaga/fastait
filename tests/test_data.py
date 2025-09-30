import torch
import fastait.data

def test_extract_cooling():
    # Create a test tensor with a clear peak and cooling events
    images = torch.tensor([
        [[1, 1], [1, 1]],  # Frame 0
        [[2, 2], [2, 2]],  # Frame 1
        [[3, 3], [3, 3]],  # Frame 2 (peak)
        [[2, 2], [2, 2]],  # Frame 3 (cooling starts)
        [[1, 1], [1, 1]],  # Frame 4
    ], dtype=torch.float32)

    cooling_images = fastait.data.extract_cooling(images)

    # Expected output is frames after the peak (frames 3 and 4)
    expected_output = torch.tensor([
        [[2, 2], [2, 2]],  # Frame 3
        [[1, 1], [1, 1]],  # Frame 4
    ], dtype=torch.float32)

    assert torch.equal(cooling_images, expected_output)

def test_normalize_percentile():
    img1 = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 20]], dtype=torch.float64)

    img2 = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 20]], dtype=torch.float64)

    images = torch.stack([img1, img2])  # shape (2, 3, 2)

    normalized_images = fastait.data.normalize_percentile(images, 0, 100)
    expected_output = (images - 1) / (20 - 1)

    assert torch.allclose(normalized_images, expected_output, atol=1e-4)