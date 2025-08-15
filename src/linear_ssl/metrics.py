from __future__ import annotations
import math
import torch
from .utils import effective_rank_from_cov, principal_angles, procrustes_align, svd_topk

@torch.no_grad()
def predictor_metrics(P: torch.Tensor, P_star: torch.Tensor):
    """
    Compare a learned predictor matrix P against the analytic optimum P_star
    in a way that is *basis/rotation invariant*. It reports:

      - W_rel_sv_err:  relative error of the *singular values* (scale-invariant)
      - W_Fro_raw:     plain Frobenius norm ||P - P_star||
      - W_Fro_aligned: Frobenius norm after optimal left/right orthogonal alignment
                       (so we ignore pure rotations of the encoder/target bases)
      - W_cos_left_min:  min cosine of principal angles between left singular subspaces
      - W_cos_right_min: min cosine of principal angles between right singular subspaces

    This is robust to the fact that BYOL-like systems are identifiable only up to
    rotations in feature space.
    """
    # Rotation-/basis-invariant SVD comparison
    U_w, S_w, Vt_w, r_w = svd_topk(P)
    U_s, S_s, Vt_s, r_s = svd_topk(P_star)
    r = min(r_w, r_s)
    # Leading subspaces
    U_w_r, U_s_r = U_w[:, :r], U_s[:, :r]          # d×r, d×r
    V_w_r, V_s_r = Vt_w[:r, :].T, Vt_s[:r, :].T    # d×r, d×r

    # (a) Singular values (basis-invariant)
    S_rel = torch.norm(S_w[:r] - S_s[:r]) / (torch.norm(S_s[:r]) + 1e-12)

    # (b) Principal angles between left/right subspaces
    cos_left  = torch.linalg.svdvals(U_w_r.T @ U_s_r).clamp(0, 1)
    cos_right = torch.linalg.svdvals(V_w_r.T @ V_s_r).clamp(0, 1)

    # (c) Two-sided Procrustes-like alignment
    # Left rotation aligns U_w -> U_s; right rotation aligns V_w -> V_s.
    R_t = U_s_r @ U_w_r.T            # d×d
    R_o = V_s_r @ V_w_r.T            # d×d   (FIXED: order was wrong before)

    fro_raw = torch.norm(P - P_star)
    P_aligned = R_t.T @ P @ R_o
    fro_aligned = torch.norm(P_aligned - P_star)

    # Robustness for r==0 (shouldn't happen, but just in case)
    if r <= 0:
        S_rel = torch.tensor(float('nan'), device=P.device)
        cos_left = cos_right = torch.tensor([float('nan')], device=P.device)

    return {
        "W_rel_sv_err": float(S_rel.item()),
        "W_Fro_raw": float(fro_raw.item()),
        "W_Fro_aligned": float(fro_aligned.item()),
        "W_cos_left_min": float(cos_left.min().item()),
        "W_cos_right_min": float(cos_right.min().item()),
    }


# src/linear_ssl/metrics.py

@torch.no_grad()
def encoder_metrics(A: torch.Tensor, B_o: torch.Tensor):
    """
    Evaluate how close the learned *online encoder* B_o is to the ideal inverse
    of the mixing matrix A in a linear-Gaussian toy.

    Shapes:
      - A    : (D × d)  mixing from latent z ∈ R^d to observed x ∈ R^D
      - B_o  : (d × D)  online encoder mapping x → z_hat
      - A^+  : (d × D)  pseudoinverse of A (ideal encoder when noise is isotropic)
    """
    device = A.device
    dtype  = A.dtype

    # Compare span(B_o^T) to span(A^+^T)
    A_pinv = torch.linalg.pinv(A)

    cos_t = principal_angles(B_o.T, A_pinv.T)

    # Projection matrices (QR) on the correct device
    QB, _ = torch.linalg.qr(B_o.T, mode="reduced")
    QP, _ = torch.linalg.qr(A_pinv.T, mode="reduced")
    proj_B = QB @ QB.T
    proj_P = QP @ QP.T
    proj_dist = torch.norm(proj_B - proj_P)

    # Procrustes alignment
    R = procrustes_align(B_o, A_pinv)

    I_lat = torch.eye(A.shape[1], device=device, dtype=dtype)
    BA_err_raw   = torch.norm(B_o @ A - I_lat)
    BA_err_align = torch.norm(R.T @ B_o @ A - I_lat)
    fro_align    = torch.norm(R.T @ B_o - A_pinv)

    return {
        "enc_cos_min": float(cos_t.min().item()),
        "enc_cos_mean": float(cos_t.mean().item()),
        "enc_proj_dist": float(proj_dist.item()),
        "BA_err_raw": float(BA_err_raw.item()),
        "BA_err_aligned": float(BA_err_align.item()),
        "enc_procrustes_fro": float(fro_align.item()),
    }


# src/linear_ssl/metrics.py

@torch.no_grad()
def effective_ranks_from_samples(Z: torch.Tensor, P: torch.Tensor, eps: float = 1e-12):
    """
    Robust effective rank via SVD of the centered data matrices.
    Works even when the sample covariance is ill-conditioned.
    erank = exp( H(p) ), p_i = λ_i / sum λ_i, λ_i = singular_values(Xc)^2 / (n-1)
    """
    def erank_from_data(X: torch.Tensor) -> float:
        n = X.shape[0]
        if n <= 1:
            return float("nan")
        # center
        Xc = X - X.mean(dim=0, keepdim=True)
        # do SVD in float64 on CPU for extra stability (50k x 10 is fine on CPU)
        Xc64 = Xc.detach().to(dtype=torch.float64, device="cpu")
        s = torch.linalg.svdvals(Xc64)          # singular values
        evals = (s ** 2) / (n - 1)              # eigenvalues of covariance
        evals = torch.clamp(evals, min=eps)
        q = evals / (evals.sum() + eps)
        H = -(q * torch.log(q)).sum()
        return float(torch.exp(H).item())

    return {
        "erank_encoded": erank_from_data(Z),
        "erank_pred":    erank_from_data(P),
    }

