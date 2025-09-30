import torch
from fastait.pct import pct

def test_pct_shape():
    """Test output shapes of PCT"""
    N, H, W = 10, 64, 64
    images = torch.randn(N, H, W, dtype=torch.float32)

    # Full SVD
    result_full = pct(images)
    assert result_full.shape[0] <= N
    assert result_full.shape[1:] == (H, W)
    assert torch.isfinite(result_full).all()

    # Partial SVD
    n_components = 5
    result_partial = pct(images, n_components=n_components)
    expected_components = min(n_components, N, H*W)
    assert result_partial.shape[0] == expected_components
    assert result_partial.shape[1:] == (H, W)
    assert torch.isfinite(result_partial).all()

def test_pct_orthogonality():
    """Test orthonormality of components"""
    N, H, W = 10, 32, 32
    images = torch.randn(N, H, W, dtype=torch.float32)
    result = pct(images, n_components=5)

    # Flatten components
    components = result.reshape(result.shape[0], -1)
    prod = components @ components.T
    assert torch.allclose(prod, torch.eye(components.shape[0]), atol=1e-5)

