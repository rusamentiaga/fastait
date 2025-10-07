import torch
import fastait.validate

def polyfit(t: torch.Tensor, y: torch.Tensor, degree: int):
    """
    Vectorized polynomial fit using Vandermonde matrix and least squares.

    Args:
        t (torch.Tensor): (N,) or (N, 1), independent variable.
        y (torch.Tensor): (N, H, W) or (N, ...), dependent variable.
        degree (int): Polynomial degree.

    Returns:
        torch.Tensor: (degree+1, H * W), polynomial coefficients in ascending order.
    """
    if t.ndim > 1:
        t = t.squeeze(-1)

    N = t.shape[0]
    X = torch.vander(t, degree + 1, increasing=True)  # (N, degree+1)
    Y = y.reshape(N, -1)  # (N, H*W)

    # Solve for coeffs using the Moore–Penrose pseudoinverse: faster but unstable
    # X_pinv = torch.linalg.pinv(X)  # (degree+1, N)
    # coeffs = X_pinv @ Y

    coeffs = torch.linalg.lstsq(X, Y).solution

    return X, coeffs

def polyval(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Evaluate polynomials at points x for all flattened pixels.

    Args:
        coeffs: (degree+1, H*W) polynomial coefficients in ascending order
        x: (N,) points to evaluate

    Returns:
        Tensor of shape (N, H*W) with polynomial values at each x
    """
    degree_plus1, num_pixels = coeffs.shape
    N = x.shape[0]

    # Expand x and coeffs for broadcasting
    x_exp = x.view(N, 1, 1)               # (N,1,1)
    powers = torch.arange(degree_plus1, device=coeffs.device, dtype=coeffs.dtype).view(1, degree_plus1, 1)  # (1,degree+1,1)
    coeffs_exp = coeffs.unsqueeze(0)      # (1, degree+1, H*W)

    # Compute x^powers: (N, degree+1, 1)
    x_powers = x_exp ** powers            # (N, degree+1,1)

    # Multiply and sum over degree dimension
    y = (coeffs_exp * x_powers).sum(dim=1) # (N,H*W)

    return y

def polyder(coeffs: torch.Tensor, order: int = 1) -> torch.Tensor:
    """
    Vectorized polynomial derivative (ascending coeff order).
    
    Args:
        coeffs (torch.Tensor): (..., degree+1), coeffs[..., i] is a_i.
        order (int): Derivative order.
    
    Returns:
        torch.Tensor: (..., degree+1-order)
    """
    deg = coeffs.shape[-1] - 1
    if order > deg:
        return torch.zeros_like(coeffs[..., :1])  # derivative vanishes

    # Powers for derivative multipliers
    i = torch.arange(order, deg+1, device=coeffs.device, dtype=coeffs.dtype)  # [order, ..., deg]

    # Falling factorial i*(i-1)*...*(i-order+1)
    factors = torch.exp(torch.lgamma(i+1) - torch.lgamma(i - order + 1))  # safe factorial ratio

    return coeffs[..., order:] * factors

class TSRResult:
    """
    Result of Thermographic Signal Reconstruction (TSR).
    Provides methods for reconstruction and derivatives.
    """
    def __init__(self, X, coeffs, H, W, log_fit):
        """
        X: (N, degree+1)
        coeffs: (degree+1, H*W)
        H, W: original image height and width
        """
        self.X = X
        self.coeffs = coeffs
        self.H = H
        self.W = W
        self.N = X.shape[0]
        self.log_fit = log_fit

    def reconstruction(self):
        """
        Reconstructed images in original domain
        """
        reconstruction = (self.X @ self.coeffs).reshape(self.N, self.H, self.W)
        if self.log_fit:
            return torch.exp(reconstruction)
        else:
            return reconstruction
    
    def derivative(self, order=1):
        """
        Derivative
        """
        coeffs_der_flat = polyder(self.coeffs.T, order=order).T
        return (self.X[:, :coeffs_der_flat.shape[0]] @ coeffs_der_flat).reshape(self.N, self.H, self.W)
    
    def first_derivative(self):
        """ 
        First derivative
        """
        return self.derivative(order=1)
    
    def second_derivative(self):
        """
        Second derivative
        """
        return self.derivative(order=2)
    
    def coefficients(self):
        """
        Polynomial coefficients in original image shape (N, H, W)
        """
        return self.coeffs.reshape(self.N, self.H, self.W)

def tsr(images: torch.Tensor, log_fit = True, degree: int = 5):
    """
    Perform Thermographic Signal Reconstruction (TSR) on a stack of images.
    Args:
        images (torch.Tensor): Input tensor of shape (N, H, W)
        log_fit (bool): Whether to fit in log domain
        degree (int): Polynomial degree for fitting in log domain
    Returns:
        TSRResult: Object containing fit results and methods for reconstruction and derivatives
    """
    fastait.validate.validate_images(images)
    N, H, W = images.shape

    # Time axis in log domain (1..N)
    t = torch.arange(1, N + 1, device=images.device, dtype=images.dtype)

    if log_fit:
        t  = torch.log(t)
        eps = torch.finfo(images.dtype).eps
        images = torch.log(torch.clamp(images, min=eps))

    # Polynomial fit
    X, coeffs_flat = polyfit(t, images, degree)  # X: (N, degree+1), coeffs_flat: (degree+1, H*W)
    return TSRResult(X, coeffs_flat, H, W, log_fit)