from __future__ import annotations
import argparse, os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ------------------
# Loading utilities
# ------------------
def load_runs(runs_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(runs_dir, "*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet in {runs_dir}")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # Normalize types (new schema only)
    for col in ["lr_pred", "lr_enc", "ema_m", "tau", "loss", "Hmax", "W_Fro_aligned",
                "erank_encoded", "erank_pred"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["epoch", "batch_size", "seed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce", downcast="integer")
    return df


# ---------------------------------------------
# Selection: value at epoch OR last valid <= T
# ---------------------------------------------
def select_at_epoch(df: pd.DataFrame, metric: str, epoch: int, lr_enc_mode: str, is_encoder_metric: bool) -> pd.DataFrame:
    """
    Take exactly 'epoch', then average across seeds.
    We always sweep on predictor LR (alpha = lr_pred) for the Y axis.
    In auto mode we collapse to a single row (alpha ignored in the grid).
    """
    sub = df[df["epoch"] == epoch].copy()
    return _finalize_selection(sub, metric, lr_enc_mode, is_encoder_metric)


def select_last_valid(df: pd.DataFrame, metric: str, epoch: int | None, lr_enc_mode: str, is_encoder_metric: bool) -> pd.DataFrame:
    """
    For each (bs,seed,alpha,tau) pick the last epoch <= epoch (if given)
    whose 'metric' is finite; then average across seeds.
    """
    sub = df.dropna(subset=[metric]).copy()
    sub = sub[np.isfinite(sub[metric].to_numpy())]
    if epoch is not None and "epoch" in sub.columns:
        sub = sub[sub["epoch"] <= epoch]

    if len(sub) == 0:
        return pd.DataFrame(columns=["batch_size", "lr_used", "ema_m", "tau", metric])

    # choose last valid per (bs, seed, alpha, tau)
    keys = [k for k in ["batch_size", "seed", "lr_pred", "tau"] if k in sub.columns]
    idx = sub.groupby(keys)["epoch"].idxmax()
    picked = sub.loc[idx].copy()

    return _finalize_selection(picked, metric, lr_enc_mode, is_encoder_metric)


def _finalize_selection(df: pd.DataFrame, metric: str, lr_enc_mode: str, is_encoder_metric: bool) -> pd.DataFrame:
    """
    Attach lr_used for pivoting (Y axis) and average across seeds.
    - same/ratio: Y axis = predictor LR (alpha = lr_pred)
    - auto:       collapse to a single row (constant lr_used=1.0)
    """
    out = df.copy()
    if lr_enc_mode == "auto":
        out["lr_used"] = 1.0  # one row
    else:
        out["lr_used"] = out["lr_pred"]  # sweep is along alpha

    group_keys = ["batch_size", "lr_used", "ema_m", "tau"]
    agg = out.groupby(group_keys, as_index=False)[metric].mean()
    return agg


# -----------------
# Plotting helpers
# -----------------
def _stable_key(x: float, nd: int = 8) -> float:
    """Round to a reproducible sci-notation float key for pivot indices."""
    try:
        return float(f"{float(x):.{nd}e}")
    except Exception:
        return np.nan


def plot_heatmap(sub: pd.DataFrame,
                 metric: str,
                 batch_size: int,
                 title: str,
                 out_path: str,
                 log_color: bool,
                 y_label: str = "learning rate (log ticks)",
                 xticks_override: list[str] | None = None,
                 eps: float = 1e-12):
    sdf = sub[sub["batch_size"] == batch_size].copy()
    if sdf.empty:
        print(f("[warn] no data for bs={batch_size} / metric={metric}"))
        return

    sdf["lr_k"] = sdf["lr_used"].map(_stable_key)
    sdf["tau_k"] = sdf["tau"].map(_stable_key)

    grid = sdf.pivot(index="lr_k", columns="tau_k", values=metric).sort_index()
    M = grid.values.astype(float)

    valid = np.isfinite(M)
    M = np.where(valid, M, np.nan)
    Mplot = np.log10(np.clip(M, eps, None)) if log_color else M

    fig, ax = plt.subplots(figsize=(8, 6))
    ma = np.ma.array(Mplot, mask=np.isnan(Mplot))
    im = ax.imshow(ma, origin="lower", aspect="auto", interpolation="nearest", cmap="viridis")
    im.cmap.set_bad(color="white")

    ax.set_title(title)

    ax.set_xticks(np.arange(len(grid.columns)))
    if xticks_override is not None:
        ax.set_xticklabels(xticks_override, rotation=45, ha="right")
    else:
        ax.set_xticklabels([f"{c:.1e}" for c in grid.columns], rotation=45, ha="right")
    ax.set_xlabel("tau = 1 - momentum (log ticks)")

    ax.set_yticks(np.arange(len(grid.index)))
    if len(grid.index) == 1 and np.isclose(grid.index[0], 1.0):
        ax.set_yticklabels(["auto"])
    else:
        ax.set_yticklabels([f"{r:.1e}" for r in grid.index])
    ax.set_ylabel(y_label)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(("log10 " if log_color else "") + metric)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def best_over_time(df: pd.DataFrame, metric: str, lr_enc_mode: str) -> pd.DataFrame:
    sub = df.dropna(subset=[metric]).copy()
    sub = sub[np.isfinite(sub[metric].to_numpy())]
    # sweep key:
    if lr_enc_mode == "auto":
        sub["lr_used"] = 1.0
    else:
        sub["lr_used"] = sub["lr_pred"]
    keys = [c for c in ["batch_size", "lr_used", "ema_m", "tau", "seed"] if c in sub.columns]
    best = sub.groupby(keys, as_index=False)[metric].min().rename(columns={metric: f"best_{metric}"})
    agg_keys = [k for k in keys if k != "seed"]
    best = best.groupby(agg_keys, as_index=False)[f"best_{metric}"].mean()
    return best


def plot_scatter_best(best: pd.DataFrame, metric: str, batch_size: int, out_path: str):
    sdf = best[best["batch_size"] == batch_size].copy()
    if sdf.empty:
        print(f"[warn] no data for bs={batch_size}]")
        return
    x = sdf["lr_used"].values
    y = sdf["tau"].values
    z = sdf[f"best_{metric}"].values

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(x, y, c=np.log10(z + 1e-12), s=40, cmap="viridis")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("predictor learning rate α (log)")
    ax.set_ylabel("tau (1 - momentum, log)")
    cbar = fig.colorbar(sc, ax=ax); cbar.set_label(f"log10 best {metric}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------- loss slices (always on) ----------
def plot_loss_slices_vs_alpha(df_raw: pd.DataFrame,
                              df_sel: pd.DataFrame,
                              bs: int,
                              epoch: int | None,
                              lr_enc_mode: str,
                              out_path: str):
    # Build per-τ curves: loss vs α (lr_pred)
    d = df_raw[df_raw["batch_size"] == bs].copy()
    d = d.dropna(subset=["tau", "lr_pred", "loss"])
    if d.empty:
        print(f"[warn] no data for loss slices (bs={bs})")
        return
    if epoch is not None and "epoch" in d.columns:
        d = d[d["epoch"] <= epoch]

    # last-valid per (tau, lr_pred, seed)
    keys = [k for k in ["tau", "lr_pred", "seed"] if k in d.columns]
    idx = d.groupby(keys)["epoch"].idxmax()
    d = d.loc[idx]

    taus = sorted(d["tau"].unique().tolist())

    fig, ax = plt.subplots(figsize=(7, 5))
    for t in taus:
        dd = d[d["tau"] == t].groupby("lr_pred", as_index=False)["loss"].median().sort_values("lr_pred")
        if dd.empty:
            continue
        ax.plot(dd["lr_pred"], dd["loss"], marker="o", ms=4, lw=1.5, label=f"τ={t:.1e}")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("predictor learning rate α")
    ax.set_ylabel("loss")
    ax.set_title(f"loss slices vs α @ epoch ≤ {epoch if epoch is not None else 'end'} (bs={bs})")
    ax.legend(ncol=2, fontsize=8, frameon=True)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# -------------
# Main program
# -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs_parquet")
    ap.add_argument("--lr_enc_mode", type=str, choices=["same", "ratio", "auto"], default="same",
                    help="how β was set in the sweep (affects Y axis and titles)")
    ap.add_argument("--metric", type=str, default="W_Fro_aligned",
                    help="predictor metric, e.g. W_Fro_aligned, W_rel_sv_err, loss")
    ap.add_argument("--enc_metric", type=str, default=None,
                    help="encoder metric; if None, auto-pick among enc_proj_dist, BA_err_aligned, enc_procrustes_fro")
    ap.add_argument("--epoch", type=int, default=1000, help="reference epoch")
    ap.add_argument("--use_last_valid", action="store_true",
                    help="use last finite value <= epoch per run before averaging (recommended)")
    ap.add_argument("--out", type=str, default="fig_heatmaps")
    ap.add_argument("--batches", type=int, nargs="*", default=None, help="subset of batch sizes to plot")
    ap.add_argument("--scatter_best", action="store_true")
    args = ap.parse_args()

    df = load_runs(args.runs_dir)

    batches = sorted(df["batch_size"].unique().tolist()) if args.batches is None else args.batches
    selector = (select_last_valid if args.use_last_valid else select_at_epoch)

    # ----- helper: ratio/alpha annotations for titles/ticks -----
    def ratio_text():
        if args.lr_enc_mode == "same":
            return "β/α = 1"
        if args.lr_enc_mode == "ratio":
            if "lr_enc" in df.columns and "lr_pred" in df.columns:
                r = np.nanmedian((df["lr_enc"] / df["lr_pred"]).to_numpy())
                return f"β/α ≈ {r:g}"
            return "β/α (ratio)"
        if args.lr_enc_mode == "auto":
            return "auto α,β from curvature"
        return ""

    def auto_alpha_xticks_for_tau(sub_df: pd.DataFrame, epoch: int | None):
        """
        For auto mode: return tick labels 'tau\n(α≈...)' with α the median lr_pred per tau.
        """
        d = sub_df.copy()
        if epoch is not None and "epoch" in d.columns:
            d = d[d["epoch"] <= epoch]
        g = d.groupby("tau")["lr_pred"].median()
        labels = []
        for t in sorted(g.index.values):
            labels.append(f"{t:.1e}\n(α≈{g.loc[t]:.1e})")
        return labels

    # 1) Predictor heatmap
    pred_sub = selector(df, args.metric, args.epoch, args.lr_enc_mode, is_encoder_metric=False)
    for bs in batches:
        title = f"{args.metric} @ epoch {args.epoch} (bs={bs}) • {ratio_text()}"
        out_path = os.path.join(args.out, f"heatmap_{args.metric}_bs{bs}_epoch{args.epoch}.png")
        xticks_override = None
        y_label = "predictor learning rate α (log ticks)"
        if args.lr_enc_mode == "auto":
            xticks_override = auto_alpha_xticks_for_tau(df[df["batch_size"] == bs], args.epoch)
            y_label = "automated learning rate"  # single row
        plot_heatmap(pred_sub, args.metric, bs, title, out_path, log_color=True,
                     y_label=y_label, xticks_override=xticks_override)

    # 2) Encoder heatmap (auto-pick metric if needed)
    enc_metric = args.enc_metric
    if enc_metric is None:
        for cand in ["enc_proj_dist", "BA_err_aligned", "enc_procrustes_fro"]:
            if cand in df.columns:
                enc_metric = cand
                break
    if enc_metric is not None and enc_metric in df.columns:
        enc_sub = selector(df, enc_metric, args.epoch, args.lr_enc_mode, is_encoder_metric=True)
        for bs in batches:
            title = f"{enc_metric} @ epoch {args.epoch} (bs={bs}) • {ratio_text()}"
            out_path = os.path.join(args.out, f"heatmap_{enc_metric}_bs{bs}_epoch{args.epoch}.png")
            xticks_override = None
            y_label = "predictor learning rate α (log ticks)"
            if args.lr_enc_mode == "auto":
                xticks_override = auto_alpha_xticks_for_tau(df[df["batch_size"] == bs], args.epoch)
                y_label = "automated learning rate"
            plot_heatmap(enc_sub, enc_metric, bs, title, out_path, log_color=True,
                         y_label=y_label, xticks_override=xticks_override)
    else:
        print("[info] no encoder metric found; skip encoder heatmap.")

    # 3) Optional scatter of best over time (predictor metric)
    if args.scatter_best:
        best = best_over_time(df, args.metric, args.lr_enc_mode)
        for bs in batches:
            out_path = os.path.join(args.out, f"scatter_best_{args.metric}_bs{bs}.png")
            plot_scatter_best(best, args.metric, bs, out_path)

    # 1b) Loss heatmap (always on)
    loss_sub = selector(df, "loss", args.epoch, args.lr_enc_mode, is_encoder_metric=False)
    for bs in batches:
        title = f"loss @ epoch {args.epoch} (bs={bs}) • {ratio_text()}"
        out_path = os.path.join(args.out, f"heatmap_loss_bs{bs}_epoch{args.epoch}.png")
        xticks_override = None
        y_label = "predictor learning rate α (log ticks)"
        if args.lr_enc_mode == "auto":
            xticks_override = auto_alpha_xticks_for_tau(df[df["batch_size"] == bs], args.epoch)
            y_label = "automated learning rate"  # single row
        plot_heatmap(
            loss_sub, "loss", bs, title, out_path,
            log_color=True, y_label=y_label, xticks_override=xticks_override
        )

        # 1c) Noise floor slices (always on): loss vs α for each τ
        out_path_slices = os.path.join(args.out, f"slices_loss_vs_alpha_bs{bs}_epoch{args.epoch}.png")
        plot_loss_slices_vs_alpha(df_raw=df, df_sel=loss_sub, bs=bs, epoch=args.epoch,
                                  lr_enc_mode=args.lr_enc_mode, out_path=out_path_slices)

    # 4) Effective-rank heatmaps (always on if available; no new flags)
    for rank_metric in ["erank_encoded", "erank_pred"]:
        if rank_metric in df.columns:
            rank_sub = selector(df, rank_metric, args.epoch, args.lr_enc_mode, is_encoder_metric=False)
            for bs in batches:
                title = f"{rank_metric} @ epoch {args.epoch} (bs={bs}) • {ratio_text()}"
                out_path = os.path.join(args.out, f"heatmap_{rank_metric}_bs{bs}_epoch{args.epoch}.png")
                xticks_override = None
                y_label = "predictor learning rate α (log ticks)"
                if args.lr_enc_mode == "auto":
                    xticks_override = auto_alpha_xticks_for_tau(df[df["batch_size"] == bs], args.epoch)
                    y_label = "automated learning rate"
                # ranks are bounded and interpretable on linear scale → no log color
                plot_heatmap(rank_sub, rank_metric, bs, title, out_path, log_color=False,
                             y_label=y_label, xticks_override=xticks_override)

    print("[ok] wrote figures to", args.out)


if __name__ == "__main__":
    main()
