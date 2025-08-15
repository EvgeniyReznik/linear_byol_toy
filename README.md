---

# Linear BYOL Toy — Predictor as LPF, Encoder as Pseudoinverse

This project implements a simple, well-instrumented BYOL-style toy on linear–Gaussian synthetic data.

* Predictor learns the Wiener / low-pass shrinkage toward the target features.
* Online encoder is learned; target encoder is an EMA of the online encoder.
* Encoders and predictor can be linear or small MLPs.
* We log theory-driven metrics (Frobenius/SVD errors to analytic optimum, subspace angles vs. pseudoinverse, Procrustes error, effective ranks, curvature estimate) to Parquet. Weights & Biases logging is optional.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start (Windows)

The repo includes a convenience script:

```
run_conda_windows.bat <single|sweep|analyze|all>
```

* `single`: run one training job with the hyperparameters defined at the top of the bat file.
* `sweep`: grid over learning rate and EMA rate (and optionally batch size).
* `analyze`: generate heatmaps and loss slices from Parquet logs.
* `all`: run single, sweep, and analyze in sequence.

Edit the variables at the top of `run_conda_windows.bat` to your liking (explained below).

## Key ideas

* Let tau = 1 - ema\_m be the EMA “update rate” (smaller tau means higher momentum).
* The predictor is a contraction toward the Wiener filter; it benefits from alpha a bit larger than tau but not so large that noise dominates.
* A simple auto schedule uses an online curvature estimate H\_t = lambda\_max(C\_oo,t) (largest eigenvalue of the online feature covariance) and sets
  alpha\_t = c \* tau / H\_t and beta\_t in the same spirit. This keeps the “student not faster than teacher” ratio roughly constant across directions.

## Hyperparameters (all variables in the bat file)

### Mode and environment

* `MODE` (single | sweep | analyze | all)
  What the script does when you run it.

* `ENV_NAME`
  Conda environment name to activate before running.

* `OUT_DIR`
  Directory where Parquet logs are written (per step and per run).

* `FIG_DIR`
  Directory where analysis figures are saved.

### Single run defaults

* `SEED`
  Random seed for reproducibility.

* `DEVICE`
  Training device: `cuda` or `cpu`.

* `D_LATENT`
  Latent dimensionality of the synthetic source z.

* `D_OBS`
  Observed dimensionality of x = A z + noise.

* `EPOCHS`
  Number of training epochs per run.

* `LOG_EVERY`
  Logging period in steps/epochs (depending on implementation).

* `BATCH_SIZE`
  Minibatch size for SGD. Affects gradient noise.

* `LR_PRED`
  Predictor learning rate alpha.

* `LR_ENC`
  Encoder learning rate beta.

* `MOM_PRED`
  SGD momentum for the predictor optimizer (not the EMA teacher).

* `MOM_ENC`
  SGD momentum for the encoder optimizer (not the EMA teacher).

* `PRED_UPDATE_EVERY`
  Update the predictor every k steps. Use >1 to make it slower in wall-clock terms.

* `ENC_UPDATE_EVERY`
  Update the encoder every k steps.

* `EMA_M`
  Momentum of the EMA teacher (m in \[0, 1)). We define tau = 1 - m as the effective EMA rate used in sweeps.

* `SIGMA_N`
  Observation noise standard deviation in x = A z + epsilon.

* `ENC`
  Encoder architecture: `linear` or `mlp`.

* `PRED`
  Predictor architecture: `linear` or `mlp`.

* `LOG_WANDB`
  true/false. Enable or disable logging to Weights & Biases.

### Sweep settings (MODE = sweep)

* `SWEEP_SEEDS`
  Space-separated list of seeds to average over in analysis.

* `SWEEP_BATCHES`
  Space-separated list of batch sizes to include in the grid.

* `LR_PRED_LOGSPACE`
  Triplet “start stop steps” for a logspace grid of predictor LR alpha.
  Example: `1e-5 1e-2 4` will create \[1e-5, 1e-4, 1e-3, 1e-2].

* `TAU_LOGSPACE`
  Triplet “start stop steps” for a logspace grid of tau = 1 - ema\_m.
  Example: `1e-4 1e-1 4` will create \[1e-4, 1e-3, 1e-2, 1e-1].
  Note: for each tau, the script sets ema\_m = 1 - tau for the teacher.

* `LR_ENC_MODE`
  How to set encoder LR beta in the sweep. One of:

  * `same`: beta = alpha.
  * `ratio`: beta = ENC\_OVER\_PRED \* alpha.
  * `logspace`: take beta from its own independent logspace grid (see LR\_ENC\_LOGSPACE).
  * `from_tau`: beta = BETA\_FROM\_TAU\_C \* tau.
  * `auto`: ignore fixed beta and let the training loop compute beta\_t from AUTO\_BETA\_C and curvature (see “Auto schedules” below).

* `ENC_OVER_PRED`
  Used if LR\_ENC\_MODE = ratio. Scales encoder LR relative to predictor LR.

* `LR_ENC_LOGSPACE`
  Triplet “start stop steps” for an independent logspace grid of encoder LR beta (used if LR\_ENC\_MODE = logspace).

* `BETA_FROM_TAU_C`
  Used if LR\_ENC\_MODE = from\_tau. Sets beta = BETA\_FROM\_TAU\_C \* tau per configuration.

* `AUTO_ALPHA_C`
  If > 0, enable predictor auto schedule inside training.
  By default, we use alpha\_t = AUTO\_ALPHA\_C \* tau / H\_t, where H\_t is the online curvature estimate (see below). Set to 0 to disable.

* `AUTO_BETA_C`
  If > 0, enable encoder auto schedule inside training.
  By default, we use beta\_t = AUTO\_BETA\_C \* tau / H\_t. Set to 0 to disable.

* `SWEEP_MOM_PRED`
  Predictor optimizer momentum used during sweeps (separate from teacher EMA).

* `SWEEP_MOM_ENC`
  Encoder optimizer momentum used during sweeps.

* `SWEEP_PRED_UPDATE_EVERY`
  Predictor update cadence in sweeps.

* `SWEEP_ENC_UPDATE_EVERY`
  Encoder update cadence in sweeps.

### Analysis

* `ANALYSIS_METRIC`
  Primary predictor metric to visualize. Typical: `W_Fro_aligned` (Frobenius error of P vs. Wiener optimum, Procrustes aligned).

* `ANALYSIS_EPOCH`
  Reference epoch at which to take values (or “last valid <= epoch” when `--use_last_valid` is passed).

* `ENC_ANALYSIS_METRIC`
  Encoder metric to visualize. Typical: `BA_err_aligned` (Procrustes-aligned error of B\_o vs. A^+).

* `INSTALL_REQS`
  true/false. If true, pip-install `requirements.txt` inside the conda env before running.

## What H\_t means and how it is computed

* We define the online feature covariance at time t as
  C\_oo,t = E\[z\_o,t z\_o,t^T] where z\_o,t = B\_o x.
* The curvature estimate is H\_t = lambda\_max(C\_oo,t), the largest eigenvalue.
  In practice we approximate it by a few steps of power iteration on a minibatch covariance.

## Auto schedules (when AUTO\_ALPHA\_C or AUTO\_BETA\_C > 0)

* Predictor: alpha\_t = c \* tau / H\_t with c = AUTO\_ALPHA\_C.
  Intuition: the effective step along an eigen-direction of curvature h is alpha\_t \* h. Keeping alpha\_t \* h roughly proportional to tau keeps the student not faster than the teacher across directions.

* Encoder: beta\_t = c \* tau / H\_t with c = AUTO\_BETA\_C.
  Intuition: the encoder sees the high curvature operator P^T P Sigma\_x. Scaling by 1 / H\_t damps overshoot in stiff directions while the EMA teacher (tau) provides a low-pass target.

Set AUTO\_\*\_C to 0 to disable and use fixed LRs.

## CLI overview (called by the bat file)

Single run:

```bash
python -m linear_ssl.train \
  --seed SEED \
  --device DEVICE \
  --d_latent D_LATENT \
  --d_obs D_OBS \
  --epochs EPOCHS \
  --log_every LOG_EVERY \
  --batch_size BATCH_SIZE \
  --lr_pred LR_PRED \
  --lr_enc LR_ENC \
  --mom_pred MOM_PRED \
  --mom_enc MOM_ENC \
  --pred_update_every PRED_UPDATE_EVERY \
  --enc_update_every ENC_UPDATE_EVERY \
  --ema_m EMA_M \
  --sigma_n SIGMA_N \
  --out_dir OUT_DIR \
  --run_name NAME \
  --enc ENC \
  --pred PRED \
  --auto_alpha_c AUTO_ALPHA_C \
  --auto_beta_c AUTO_BETA_C \
  --log_wandb LOG_WANDB
```

Sweep:

```bash
python scripts/run_sweep.py \
  --out_dir OUT_DIR \
  --seeds SWEEP_SEEDS \
  --batch_sizes SWEEP_BATCHES \
  --lr_pred_logspace LR_PRED_LOGSPACE \
  --tau_logspace TAU_LOGSPACE \
  --lr_enc_mode LR_ENC_MODE \
  --enc_over_pred ENC_OVER_PRED \
  --lr_enc_logspace LR_ENC_LOGSPACE \
  --beta_from_tau_c BETA_FROM_TAU_C \
  --pred_update_every SWEEP_PRED_UPDATE_EVERY \
  --enc_update_every SWEEP_ENC_UPDATE_EVERY \
  --mom_pred SWEEP_MOM_PRED \
  --mom_enc SWEEP_MOM_ENC \
  --auto_alpha_c AUTO_ALPHA_C \
  --auto_beta_c AUTO_BETA_C \
  --epochs EPOCHS \
  --log_every LOG_EVERY \
  --sigma_n SIGMA_N \
  --d_latent D_LATENT \
  --d_obs D_OBS \
  --device DEVICE \
  --enc ENC \
  --pred PRED \
  --log_wandb LOG_WANDB
```

Analysis:

```bash
python analysis/heatmaps.py \
  --runs_dir OUT_DIR \
  --lr_enc_mode LR_ENC_MODE \
  --metric ANALYSIS_METRIC \
  --epoch ANALYSIS_EPOCH \
  --out FIG_DIR \
  --use_last_valid \
  --enc_metric ENC_ANALYSIS_METRIC
```

## Metrics (what to look at)

* `W_Fro_aligned`: Frobenius error between P and the Wiener optimum P\* = C\_to C\_oo^{-1}, after Procrustes alignment (R-aligned).
* `BA_err_aligned`: Procrustes-aligned error between B\_o and A^+ (pseudo-inverse).
* `erank_pred`, `erank_encoded`: effective rank of predicted and encoded features (collapse diagnostics).
* `Hmax`: curvature estimate H\_t = lambda\_max(C\_oo,t).
* `loss`: training loss. We also plot loss vs. alpha slices across tau to show the U-shape.

## Tips

* EMA: tau = 1 - ema\_m. High momentum (ema\_m close to 1) means small tau.
* Good predictor regions often have alpha a bit larger than tau but not so large that noise dominates. The auto schedule helps.
* For encoder stability, prefer higher momentum (small tau) and moderate or small beta.

---
