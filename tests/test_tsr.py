import torch
import numpy as np
import fastait.tsr

def test_polyfit():
    N, H, W, degree = 10, 4, 4, 3

    # Random positive images
    np.random.seed(0)
    images_np = np.random.rand(N, H, W).astype(np.float32) + 1e-2

    # Torch version
    t_torch = torch.arange(1, N + 1, dtype=torch.float32)
    images_torch = torch.tensor(images_np)
    _, coeffs_torch = fastait.tsr.polyfit(t_torch, images_torch, degree)  # (degree+1, H * W)

    # NumPy reference using polyfit
    t_np = np.arange(1, N + 1, dtype=np.float32)
    coeffs_np = np.zeros((degree + 1, H *W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            y = images_np[:, i, j]
            p = np.polyfit(t_np, y, deg=degree)
            index = i * W + j
            coeffs_np[:, index] = p[::-1]  # ascending order

    # Compare
    assert np.allclose(coeffs_torch.numpy(), coeffs_np, rtol=1e-4, atol=1e-4)

def test_polyder():
    torch.manual_seed(0)
    np.random.seed(0)

    # Random polynomial coefficients in ascending order (a0 + a1 x + a2 x^2 + ...)
    coeffs_np = np.random.randn(5).astype(np.float32)  # degree 4
    coeffs_torch = torch.tensor(coeffs_np)

    # Test multiple derivative orders
    for order in range(1, 5):
        # Torch result (ascending order)
        coeffs_der_torch = fastait.tsr.polyder(coeffs_torch, order=order).numpy()

        # NumPy reference (descending order, so flip before and after)
        coeffs_desc = coeffs_np[::-1]              # make descending
        coeffs_der_desc = np.polyder(coeffs_desc, m=order)  # derivative in descending
        coeffs_der_np = coeffs_der_desc[::-1]      # back to ascending

        assert np.allclose(coeffs_der_torch, coeffs_der_np, rtol=1e-6, atol=1e-6), \
            f"Mismatch for order {order}"
        
def test_polyval():
    N, H, W, degree = 10, 4, 4, 2

    # Random positive images
    np.random.seed(0)
    images_np = np.random.rand(N, H, W).astype(np.float32) + 1e-2

    # Flatten pixels
    images_flat = images_np.reshape(N, H*W)

    # Random t values
    t_np = np.arange(1, N + 1, dtype=np.float32)
    t_torch = torch.tensor(t_np)

    # Fit polynomial per pixel using NumPy
    coeffs_np = np.zeros((degree+1, H*W), dtype=np.float32)
    for idx in range(H*W):
        p = np.polyfit(t_np, images_flat[:, idx], deg=degree)
        coeffs_np[:, idx] = p[::-1]  # ascending order

    coeffs_torch = torch.tensor(coeffs_np)

    y_torch = fastait.tsr.polyval(coeffs_torch, t_torch)

    # NumPy evaluation
    y_np = np.zeros_like(images_flat)
    for idx in range(H*W):
        y_np[:, idx] = np.polyval(coeffs_np[:, idx][::-1], t_np)

    # Compare
    assert np.allclose(y_torch.numpy(), y_np, rtol=1e-5, atol=1e-6)
