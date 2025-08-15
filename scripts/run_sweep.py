# scripts/run_sweep.py
from __future__ import annotations
import argparse, os, itertools, subprocess, sys, math
import numpy as np

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="runs_parquet")

    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=[32, 64])

    # grid over predictor LR (alpha) and tau = 1 - momentum
    ap.add_argument("--lr_pred_logspace", type=float, nargs=3, default=[1e-5, 1e-2, 6],
                    help="start stop steps for predictor LR α")
    ap.add_argument("--tau_logspace", type=float, nargs=3, default=[1e-4, 1e-1, 6],
                    help="start stop steps for tau=1-m")

    # encoder LR (beta): choose how to set it
    ap.add_argument("--lr_enc_mode", choices=["same","ratio","logspace","from_tau","auto"],
                    default="same",
                    help="how to set encoder LR β")
    ap.add_argument("--enc_over_pred", type=float, default=1.0,
                    help='if mode=ratio, set β = enc_over_pred * α')
    ap.add_argument("--lr_enc_logspace", type=float, nargs=3, default=[1e-5, 1e-2, 6],
                    help="if mode=logspace, start stop steps for β")
    ap.add_argument("--beta_from_tau_c", type=float, default=0.4,
                    help="if mode=from_tau, set β = c * tau (H-agnostic)")

    # optional update frequencies (further time-scale separation)
    ap.add_argument("--pred_update_every", type=int, default=1)
    ap.add_argument("--enc_update_every", type=int, default=1)

    # momentum per module (SGD momentum, not EMA)
    ap.add_argument("--mom_pred", type=float, default=0.9)
    ap.add_argument("--mom_enc",  type=float, default=0.9)

    # EMA momentum (teacher)
    ap.add_argument("--ema_m_default", type=float, default=None,
                    help="fallback momentum m if you prefer to sweep m directly; \
                          otherwise we sweep tau and set m = 1 - tau")

    # auto schedules (passed through to train; train computes α≈c*2/H, β≈c*tau/H)
    ap.add_argument("--auto_alpha_c", type=float, default=0.0)
    ap.add_argument("--auto_beta_c",  type=float, default=0.0)

    # training / data
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--sigma_n", type=float, default=0.5)
    ap.add_argument("--d_latent", type=int, default=10)
    ap.add_argument("--d_obs", type=int, default=100)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--enc", choices=["linear","mlp"], default="linear")
    ap.add_argument("--pred", choices=["linear","mlp"], default="linear")
    ap.add_argument("--log_wandb", type=str, default="false")

    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()

def logspace(lo, hi, steps):
    return [float(v) for v in np.logspace(np.log10(lo), np.log10(hi), int(steps))]

def deconflict_tau(taus, lrs_pred, lrs_enc=None, rel=1e-10, jitter=1e-6):
    """
    Nudge any tau that nearly equals α (and optionally β) to avoid exact equality
    that can poison pivots/joins or binning.
    """
    out = []
    for t in taus:
        clash = any(math.isclose(t, a, rel_tol=rel, abs_tol=0.0) for a in lrs_pred)
        if lrs_enc is not None:
            clash = clash or any(math.isclose(t, b, rel_tol=rel, abs_tol=0.0) for b in lrs_enc)
        if clash:
            t = t * (1.0 + jitter)
        out.append(t)
    return out

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Build α (predictor) grid and τ (EMA step) grid
    alphas = logspace(*args.lr_pred_logspace)
    taus   = logspace(*args.tau_logspace)

    # Build β grid depending on mode
    if args.lr_enc_mode == "same":
        betas = None  # will be tied to α per run
    elif args.lr_enc_mode == "ratio":
        betas = [args.enc_over_pred * a for a in alphas]
    elif args.lr_enc_mode == "logspace":
        betas = logspace(*args.lr_enc_logspace)
    elif args.lr_enc_mode == "from_tau":
        # we’ll compute per (alpha,tau) pair later: beta = c * tau
        betas = None
    elif args.lr_enc_mode == "auto":
        # train will compute α, β from curvature; α/β values here are irrelevant
        betas = None

    # For deconflict we need concrete lists
    lrs_pred_for_conflict = alphas
    lrs_enc_for_conflict  = betas if (betas is not None) else []

    taus = deconflict_tau(taus, lrs_pred_for_conflict, lrs_enc_for_conflict)

    # Momentum m grid from tau (unless user forces a single m)
    Ms = [1.0 - t for t in taus] if args.ema_m_default is None else [args.ema_m_default]

    # Cartesian product
    base_grid = list(itertools.product(args.seeds, args.batch_sizes, alphas, Ms))

    print(f"[grid] seeds×bs×alpha×tau  = {len(base_grid)} runs"
          + ("" if args.lr_enc_mode in ["same","from_tau","auto"] else
             f"  (β mode: {args.lr_enc_mode})"))

    for seed, bs, alpha, m in base_grid:
        tau = 1.0 - m

        # Resolve beta for this run
        if args.lr_enc_mode == "same":
            beta = alpha
        elif args.lr_enc_mode == "ratio":
            beta = args.enc_over_pred * alpha
        elif args.lr_enc_mode == "logspace":
            # loop over explicit betas as well
            for beta in betas:
                run_one(args, seed, bs, alpha, beta, m, tau)
            continue
        elif args.lr_enc_mode == "from_tau":
            beta = args.beta_from_tau_c * tau
        elif args.lr_enc_mode == "auto":
            beta = None  # train will set it

        run_one(args, seed, bs, alpha, beta, m, tau)

def run_one(args, seed, bs, alpha, beta, m, tau):
    # Compose run name
    name_bits = [
        f"seed{seed}",
        f"bs{bs}",
        f"alp{alpha:.2e}",
        f"m{m:.5f}"
    ]
    if beta is not None:
        name_bits.insert(3, f"bet{beta:.2e}")
    if args.auto_alpha_c > 0 or args.auto_beta_c > 0:
        name_bits.append(f"autoA{args.auto_alpha_c:g}_B{args.auto_beta_c:g}")
    run_name = "_".join(name_bits)

    cmd = [
        sys.executable, "-m", "linear_ssl.train",
        "--seed", str(seed),
        "--device", args.device,
        "--d_latent", str(args.d_latent),
        "--d_obs", str(args.d_obs),
        "--epochs", str(args.epochs),
        "--log_every", str(args.log_every),
        "--batch_size", str(bs),
        "--ema_m", str(m),
        "--sigma_n", str(args.sigma_n),
        "--out_dir", args.out_dir,
        "--run_name", run_name,
        "--enc", args.enc,
        "--pred", args.pred,
        "--mom_pred", str(args.mom_pred),
        "--mom_enc",  str(args.mom_enc),
        "--pred_update_every", str(args.pred_update_every),
        "--enc_update_every",  str(args.enc_update_every),
        "--log_wandb", args.log_wandb,
    ]

    # LRs: pass explicit α,β if given; otherwise rely on auto rules
    if args.auto_alpha_c > 0:
        cmd += ["--auto_alpha_c", str(args.auto_alpha_c)]
    else:
        cmd += ["--lr_pred", str(alpha)]

    if args.lr_enc_mode == "auto" and args.auto_beta_c > 0:
        cmd += ["--auto_beta_c", str(args.auto_beta_c)]
    else:
        # beta may be None if using auto without providing c; default to α to be safe
        cmd += ["--lr_enc", str(beta if beta is not None else alpha)]

    print("[run]", " ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
