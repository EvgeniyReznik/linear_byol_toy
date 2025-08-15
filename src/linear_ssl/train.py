# src/linear_ssl/train.py
from __future__ import annotations
import argparse, os, math
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .utils import set_seed, to_device, walltime_s
from .data import make_mixing_A, make_latent_spectrum, sample_batch
from .models import (
    LinearEncoder, MLPEncoder,
    LinearPredictor, MLPPredictor,
    ema_update, analytic_P_star, curvature_Hmax
)
from .metrics import predictor_metrics, encoder_metrics, effective_ranks_from_samples
from .logging_io import ParquetLogger


def make_encoder(name: str, d_obs: int, d_latent: int, A=None):
    if name == "linear":
        initW = None
        if A is not None:
            I_lat = torch.eye(d_latent, device=A.device, dtype=A.dtype)
            if torch.allclose(A.T @ A, I_lat, atol=1e-4):
                initW = A.T.clone()
        return LinearEncoder(d_obs, d_latent, init_weight=initW)
    elif name == "mlp":
        return MLPEncoder(d_obs, d_latent, hidden=256, act="tanh")
    else:
        raise ValueError("--enc must be 'linear' or 'mlp'")


def make_predictor(name: str, d_latent: int):
    if name == "linear":
        return LinearPredictor(d_latent)
    elif name == "mlp":
        return MLPPredictor(d_latent, hidden=256, act="tanh")
    else:
        raise ValueError("--pred must be 'linear' or 'mlp'")


def parse_args():
    ap = argparse.ArgumentParser()
    # data / run
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--d_latent", type=int, default=10)
    ap.add_argument("--d_obs", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--sigma_n", type=float, default=0.5)
    ap.add_argument("--out_dir", type=str, default="runs_parquet")
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--enc", type=str, choices=["linear", "mlp"], default="linear")
    ap.add_argument("--pred", type=str, choices=["linear", "mlp"], default="linear")

    # EMA (teacher) momentum
    ap.add_argument("--ema_m", type=float, default=0.999)  # m; tau = 1 - m

    # SEPARATE time-scales
    ap.add_argument("--lr_pred", type=float, default=None, help="predictor LR α (if None, falls back to --lr)")
    ap.add_argument("--lr_enc",  type=float, default=None, help="encoder LR β (if None, falls back to --lr)")
    ap.add_argument("--lr", type=float, default=1e-3, help="legacy LR; used if lr_pred/lr_enc are None")
    ap.add_argument("--mom_pred", type=float, default=0.9)
    ap.add_argument("--mom_enc",  type=float, default=0.9)

    # Optional update frequencies (for additional time-scale separation)
    ap.add_argument("--pred_update_every", type=int, default=1, help="step predictor every k iters")
    ap.add_argument("--enc_update_every",  type=int, default=1, help="step encoder every k iters")

    # Optional auto-scheduling from curvature rules (set c=0 to disable)
    ap.add_argument("--auto_alpha_c", type=float, default=0.0,
                    help="if >0, set α = c * 2/Hmax (clipped below 2/Hmax)")
    ap.add_argument("--auto_beta_c", type=float, default=0.0,
                    help="if >0, set β = c * tau / Hmax (clipped below 2/Hmax)")

    # logging
    ap.add_argument("--log_wandb", type=str, default="false")
    ap.add_argument("--wandb_project", type=str, default="linear-byol-toy")
    return ap.parse_args()


def maybe_init_wandb(args):
    use_wandb = args.log_wandb.lower() in ["1", "true", "yes", "y"]
    if not use_wandb:
        return None
    try:
        import wandb
        run = wandb.init(project=args.wandb_project,
                         config=vars(args),
                         name=args.run_name if args.run_name else None)
        return run
    except Exception as e:
        print("[warn] wandb not available:", e)
        return None


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # Synthetic data
    A = make_mixing_A(args.d_obs, args.d_latent, orthonormal=True).to(device)
    lambdas = make_latent_spectrum(args.d_latent, kind="lin", lo=0.1, hi=100.0).to(device)

    # Modules
    enc_online = make_encoder(args.enc, args.d_obs, args.d_latent, A=A).to(device)
    enc_target = make_encoder(args.enc, args.d_obs, args.d_latent, A=A).to(device)
    for p_t, p_o in zip(enc_target.parameters(), enc_online.parameters()):
        p_t.data.copy_(p_o.data)
    predictor = make_predictor(args.pred, args.d_latent).to(device)

    tau = 1.0 - args.ema_m

    # ---- Curvature estimate for auto schedules (uses predictor curvature H = λ_max(C_oo))
    with torch.no_grad():
        Hmax0 = curvature_Hmax(
            A,
            enc_online.W if args.enc == "linear" else enc_online(torch.zeros(1, args.d_obs, device=device))[:1] * 0 + 1,
            lambdas,
            args.sigma_n,
        )
        Hmax0 = float(Hmax0)

    # Resolve LRs
    lr_pred = args.lr if args.lr_pred is None else args.lr_pred
    lr_enc  = args.lr if args.lr_enc  is None else args.lr_enc

    # Optional auto α, β rules (clipped strictly below 2/H)
    if args.auto_alpha_c > 0 and Hmax0 > 0:
        lr_pred = min(args.auto_alpha_c * (2.0 / Hmax0), 0.99 * (2.0 / Hmax0))
    if args.auto_beta_c > 0 and Hmax0 > 0:
        lr_enc  = min(args.auto_beta_c * (tau / Hmax0), 0.99 * (2.0 / Hmax0))

    # Two separate optimizers (so we can step on different schedules)
    opt_pred = optim.SGD(predictor.parameters(), lr=lr_pred, momentum=args.mom_pred)
    opt_enc  = optim.SGD(enc_online.parameters(), lr=lr_enc, momentum=args.mom_enc)

    # Logging
    run_id = args.run_name or f"seed{args.seed}_bs{args.batch_size}_alp{lr_pred:.2e}_bet{lr_enc:.2e}_m{args.ema_m:.5f}"
    out_path = os.path.join(args.out_dir, f"{run_id}.parquet")
    os.makedirs(args.out_dir, exist_ok=True)
    pqlog = ParquetLogger(out_path)
    wb = maybe_init_wandb(args)

    t0 = walltime_s()
    for epoch in tqdm(range(1, args.epochs + 1), ncols=100, desc="train"):
        # sample a minibatch
        z, x, x_plus = sample_batch(A, lambdas, args.sigma_n, args.batch_size, device=device)

        # forward
        z_o = enc_online(x)
        with torch.no_grad():
            z_t = enc_target(x_plus)  # stop-grad on target
        p = predictor(z_o)

        loss = ((p - z_t) ** 2).mean()

        # zero grads for both modules each iter
        opt_pred.zero_grad(set_to_none=True)
        opt_enc.zero_grad(set_to_none=True)

        loss.backward()

        # Step on chosen schedules
        if (epoch % max(1, args.pred_update_every)) == 0:
            opt_pred.step()
        if (epoch % max(1, args.enc_update_every)) == 0:
            opt_enc.step()

        # EMA update of target encoder (every iter)
        with torch.no_grad():
            ema_update(enc_target, enc_online, args.ema_m)

        # Optionally refresh auto α/β using current curvature every log interval
        if (args.auto_alpha_c > 0 or args.auto_beta_c > 0) and (epoch % args.log_every == 0):
            with torch.no_grad():
                Hcur = curvature_Hmax(
                    A,
                    enc_online.W if args.enc == "linear" else enc_online(x)[:1] * 0 + 1,
                    lambdas,
                    args.sigma_n,
                )
                Hcur = float(Hcur) if torch.is_tensor(Hcur) else float(Hcur)
            if args.auto_alpha_c > 0 and Hcur > 0:
                new_alpha = min(args.auto_alpha_c * (2.0 / Hcur), 0.99 * (2.0 / Hcur))
                for g in opt_pred.param_groups:
                    g["lr"] = new_alpha
                lr_pred = new_alpha
            if args.auto_beta_c > 0 and Hcur > 0:
                new_beta = min(args.auto_beta_c * (tau / Hcur), 0.99 * (2.0 / Hcur))
                for g in opt_enc.param_groups:
                    g["lr"] = new_beta
                lr_enc = new_beta

        # Logging
        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            with torch.no_grad():
                # analytic P* (linear-only)
                can_analytic = (args.enc == "linear" and args.pred == "linear")
                P_star = None
                if can_analytic:
                    P_star = analytic_P_star(A, enc_online.W, enc_target.W, lambdas, args.sigma_n)

                # predictor metrics
                pred_m = predictor_metrics(predictor.W, P_star) if can_analytic else {}

                # encoder metrics
                enc_m = encoder_metrics(A, enc_online.W) if args.enc == "linear" else {}

                # effective ranks (big batch)
                z_big, x_big, _ = sample_batch(A, lambdas, args.sigma_n, batch=8192, device=device)
                z_enc = enc_online(x_big)
                p_big = predictor(z_enc)
                try:
                    eranks = effective_ranks_from_samples(z_enc, p_big)
                except Exception:
                    eranks = {"erank_encoded": float("nan"), "erank_pred": float("nan")}

                # curvature (for plotting rules/overlays)
                try:
                    Hmax = curvature_Hmax(
                        A,
                        enc_online.W if args.enc == "linear" else enc_online(x_big)[:1] * 0 + 1,
                        lambdas,
                        args.sigma_n,
                    )
                except Exception:
                    Hmax = float("nan")

                row = {
                    "epoch": int(epoch),
                    "seed": int(args.seed),
                    "batch_size": int(args.batch_size),
                    # timescales:
                    "lr_pred": float(lr_pred),
                    "lr_enc": float(lr_enc),
                    "mom_pred": float(args.mom_pred),
                    "mom_enc": float(args.mom_enc),
                    "lr": float(args.lr),             # legacy, for backward compatibility
                    "ema_m": float(args.ema_m),
                    "tau": float(tau),
                    # setup:
                    "sigma_n": float(args.sigma_n),
                    "d_latent": int(args.d_latent),
                    "d_obs": int(args.d_obs),
                    # loss & curvature:
                    "loss": float(loss.item()),
                    "Hmax": float(Hmax if not torch.is_tensor(Hmax) else Hmax.item()),
                    "wall_s": float(walltime_s() - t0),
                }
                row.update(pred_m)
                row.update(enc_m)
                row.update(eranks)

                pqlog.write_row(row)

                if wb is not None:
                    log_row = {k: v for k, v in row.items() if isinstance(v, (int, float))}
                    wb.log(log_row, step=epoch)

    pqlog.close()
    if wb is not None:
        wb.finish()
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
