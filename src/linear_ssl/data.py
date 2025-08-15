from __future__ import annotations
import torch

@torch.no_grad()
def make_mixing_A(d_obs: int, d_latent: int, orthonormal: bool = True) -> torch.Tensor:
    """
    Build a D×d mixing A. If orthonormal=True => columns orthonormal (A^T A = I).
    """
    A_raw = torch.randn(d_obs, d_latent)
    if orthonormal:
        A, _ = torch.linalg.qr(A_raw, mode='reduced')  # A^T A = I
        return A
    return A_raw  #TDOO: full column rank not guaranteed

@torch.no_grad()
def make_latent_spectrum(d_latent: int, kind: str = "lin", lo: float = 0.1, hi: float = 100.0) -> torch.Tensor:
    if kind == "lin":
        lambdas = torch.linspace(lo, hi, d_latent)
    elif kind == "log":
        lambdas = torch.logspace(torch.log10(torch.tensor(lo)), torch.log10(torch.tensor(hi)), d_latent)
    else:
        raise ValueError("kind must be 'lin' or 'log'")
    return lambdas

@torch.no_grad()
def sample_batch(A: torch.Tensor, lambdas: torch.Tensor, sigma_n: float, batch: int, device="cpu"):
    # z ~ N(0, Λ), x = A z + ε, x+ = A z + ε'
    d_latent = lambdas.shape[0]
    z = torch.randn(batch, d_latent, device=device) * torch.sqrt(lambdas.to(device))
    eps1 = torch.randn(batch, A.shape[0], device=device) * sigma_n
    eps2 = torch.randn(batch, A.shape[0], device=device) * sigma_n
    x = z @ A.T.to(device) + eps1
    x_plus = z @ A.T.to(device) + eps2
    return z, x, x_plus
