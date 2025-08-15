from __future__ import annotations
import json, math, random, time, os
import numpy as np
import torch

def set_seed(seed: int):
    # initialize random state with seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj

def effective_rank_from_cov(C: torch.Tensor, eps: float = 1e-12) -> float:
    # C must be symmetric PSD
    evals = torch.linalg.eigvalsh(C).clamp_min(eps)
    p = evals / evals.sum()
    erank = torch.exp(-(p * torch.log(p)).sum())
    return float(erank.item())

def principal_angles(Q1: torch.Tensor, Q2: torch.Tensor):
    # cosines of principal angles between column spaces of Q1 and Q2
    Q1, _ = torch.linalg.qr(Q1, mode="reduced")
    Q2, _ = torch.linalg.qr(Q2, mode="reduced")
    s = torch.linalg.svdvals(Q1.T @ Q2).clamp(0, 1)
    return s

def procrustes_align(A: torch.Tensor, B: torch.Tensor):
    # find R (orthogonal) minimizing ||R^T A - B||_F
    # caution: stable if no multiple same singular values
    U, _, Vt = torch.linalg.svd(A @ B.T)
    R = U @ Vt
    return R

def svd_topk(M: torch.Tensor, eps: float = 1e-8):
    # relative rank estimation and truncated SVD helper
    U, S, Vt = torch.linalg.svd(M)
    r = int((S > eps * S.max()).sum().item())
    r = max(r, 1)
    return U[:, :r], S[:r], Vt[:r, :], r

def json_dumps(obj) -> str:
    try:
        return json.dumps(obj)
    except Exception:
        return json.dumps(str(obj))

def walltime_s():
    return time.time()
