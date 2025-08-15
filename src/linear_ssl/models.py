from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional

class LinearEncoder(nn.Module):
    def __init__(self, d_obs: int, d_latent: int, init_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.linear = nn.Linear(d_obs, d_latent, bias=False)
        if init_weight is not None:
            with torch.no_grad():
                self.linear.weight.copy_(init_weight)

    def forward(self, x):
        return self.linear(x)

    @property
    def W(self):
        return self.linear.weight

class MLPEncoder(nn.Module):
    def __init__(self, d_obs: int, d_latent: int, hidden: int = 256, act: str = "tanh"):
        super().__init__()
        act_layer = nn.Tanh() if act == "tanh" else nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(d_obs, hidden, bias=True),
            act_layer,
            nn.Linear(hidden, d_latent, bias=False),
        )
    def forward(self, x): return self.net(x)

class LinearPredictor(nn.Module):
    def __init__(self, d_latent: int):
        super().__init__()
        self.linear = nn.Linear(d_latent, d_latent, bias=False)
    def forward(self, z): return self.linear(z)
    @property
    def W(self): return self.linear.weight

class MLPPredictor(nn.Module):
    def __init__(self, d_latent: int, hidden: int = 256, act: str = "tanh"):
        super().__init__()
        act_layer = nn.Tanh() if act == "tanh" else nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(d_latent, hidden), act_layer,
            nn.Linear(hidden, d_latent, bias=False),
        )
    def forward(self, z): return self.net(z)

@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, m: float):
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.data.mul_(m).add_(p_o.data, alpha=1.0 - m)

@torch.no_grad()
def analytic_P_star(A: torch.Tensor, B_o: torch.Tensor, B_t: torch.Tensor,
                    lambdas: torch.Tensor, sigma_n: float) -> torch.Tensor:
    """
    Closed-form population-optimal predictor for linear BYOL.

    We minimize the population loss ½ E[ || P z_o - z_t ||² ] with stop-grad on z_t.
    With z_o = B_o x and z_t = B_t x⁺, where the two views share signal but have
    independent noise, the standard linear regression/Wiener solution is

        P* = C_to C_oo^{-1}

    where
        C_oo = E[z_o z_oᵀ]  = B_o (A Λ Aᵀ + σ² I) B_oᵀ
        C_to = E[z_t z_oᵀ]  = B_t (A Λ Aᵀ)        B_oᵀ

    Shapes:
        A      : (D x d)
        B_o,B_t: (d x D)
        Λ      : (d x d)
        C_oo   : (d x d)  symmetric PSD
        C_to   : (d x d)

    Notes on numerics:
      * We use pinv(C_oo) to guard against rank-deficiency/ill-conditioning
        (common early in training or with small batch estimates).
      * If you *know* C_oo ≻ 0 and well-conditioned, you can replace pinv with a
        Cholesky solve for speed/accuracy:
            P_star = torch.cholesky_solve(C_to.T, torch.cholesky(C_oo)).T
      * For extra robustness you may symmetrize C_oo and add a tiny ridge:
            C_oo = 0.5*(C_oo + C_oo.T) + eps*I
    """
    # P* = C_to C_oo^{-1} with:
    # C_oo = B_o A Λ A^T B_o^T + σ^2 B_o B_o^T
    # C_to = B_t A Λ A^T B_o^T
    Sig = A @ torch.diag(lambdas) @ A.T
    C_oo = B_o @ Sig @ B_o.T + (sigma_n ** 2) * (B_o @ B_o.T)
    C_to = B_t @ Sig @ B_o.T
    return C_to @ torch.linalg.pinv(C_oo)  # pinv guards near-singular cases

@torch.no_grad()
def curvature_Hmax(A: torch.Tensor, B_o: torch.Tensor, lambdas: torch.Tensor, sigma_n: float) -> float:
    """
    Return the largest eigenvalue (curvature) of the online feature covariance:
        C_oo = E[z_o z_oᵀ] = B_o (A Λ Aᵀ + σ_n^2 I_D) B_oᵀ.

    Why this matters:
      • For predictor GD with fixed encoders, the linearized update is
            P_{t+1} = P_t - α (P_t C_oo - C_to),
        so the contraction factor along the worst mode is controlled by
        λ_max(C_oo). A safe deterministic step-size bound is
            0 < α < 2 / λ_max(C_oo).
      • In the two-time-scale (EMA) setting, a useful rule is
            (α * λ_max(C_oo)) / τ  ≲  1,  where τ = 1 - m.

    Shapes:
      A      : (D x d)
      B_o    : (d x D)
      Λ      : (d x d) via diag(lambdas)
      C_oo   : (d x d), symmetric PSD
      return : Python float (λ_max)

    Notes on numerics:
      • We form Σ_sig = A Λ Aᵀ once, then C_oo = B_o Σ_sig B_oᵀ + σ_n² B_o B_oᵀ.
      • eigvalsh expects a symmetric input; if you worry about tiny asymmetries,
        you can symmetrize: C_oo = 0.5*(C_oo + C_oo.T).
      • For very large d, consider a few steps of power iteration instead of
        a full eigen decomposition.
    """
    Sig = A @ torch.diag(lambdas) @ A.T
    C_oo = B_o @ Sig @ B_o.T + (sigma_n ** 2) * (B_o @ B_o.T)
    Hmax = torch.linalg.eigvalsh(C_oo).max().item()
    return float(Hmax)

