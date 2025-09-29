import torch
import pytest

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
def test_gpu():
    assert torch.cuda.is_available() == True

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
def test_pytorch_tensors():
    # Create a tensor on the GPU
    tensor = torch.tensor([1, 2, 3]).cuda()
    assert tensor.device.type == 'cuda'

    # Create a tensor on the CPU
    tensor = torch.tensor([1, 2, 3])
    assert tensor.device.type == 'cpu'

    # Move a tensor to the GPU
    tensor = torch.tensor([1, 2, 3]).to('cuda')
    assert tensor.device.type == 'cuda'

    # Move a tensor to the CPU
    tensor = torch.tensor([1, 2, 3]).to('cpu')
    assert tensor.device.type == 'cpu'

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
def test_pytorch_tensor_operations():
    # Create two tensors on the GPU
    tensor1 = torch.tensor([1, 2, 3]).cuda()
    tensor2 = torch.tensor([4, 5, 6]).cuda()

    # Perform an operation on the tensors
    result = tensor1 + tensor2
    assert torch.equal(result, torch.tensor([5, 7, 9]).cuda())