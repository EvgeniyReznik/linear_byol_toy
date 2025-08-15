---

# Linear BYOL Toy

A small BYOL-style learning toy on linear Gaussian synthetic data.
The predictor learns a Wiener (low-pass) shrinkage, the online encoder is learned,
and the target encoder is an EMA of the online encoder. Encoders and predictor can
be linear or small MLPs. Runs log theory-driven metrics to Parquet; Weights & Biases
is optional.

## Project Structure

### Code Directories
- src/
- analysis/
- scripts/

### Entry Points
- train: linear_ssl.train
- sweep: scripts/run_sweep.py
- analysis: analysis/heatmaps.py

### Output Directories
- parquet_runs: runs_parquet
- figures: figs

## Installation

```
# Recommended: use a clean Python or conda environment
python -m venv .venv
# On Unix/macOS
source .venv/bin/activate
# On Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Quick Start

Windows convenience script (inside an activated conda env):
  run_conda_windows.bat <single|sweep|analyze|all>

Examples:
  run_conda_windows.bat single
  run_conda_windows.bat sweep
  run_conda_windows.bat analyze
  run_conda_windows.bat all

## Modes

- **single**: Run one training job with fixed hyperparameters. Writes a Parquet file per run.
- **sweep**: Grid over alpha (predictor LR), tau (EMA rate), batch size, and optional encoder LR scheme. Multiple seeds supported.
- **analyze**: Produce heatmaps and helper figures from runs_parquet into figs.
- **all**: Run single, then sweep, then analyze sequentially.

## Commands

### Training

```
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
  --out_dir runs_parquet \
  --run_name NAME \
  --enc ENC \
  --pred PRED \
  --auto_alpha_c AUTO_ALPHA_C \
  --auto_beta_c AUTO_BETA_C \
  --log_wandb LOG_WANDB
```

### Sweep

```
python scripts/run_sweep.py \
  --out_dir runs_parquet \
  --seeds <list of ints> \
  --batch_sizes <list of ints> \
  --lr_pred_logspace <start stop steps> \
  --tau_logspace <start stop steps> \
  --lr_enc_mode <same|ratio|logspace|from_tau|auto> \
  --enc_over_pred ENC_OVER_PRED \
  --lr_enc_logspace <start stop steps> \
  --beta_from_tau_c BETA_FROM_TAU_C \
  --pred_update_every PRED_UPDATE_EVERY \
  --enc_update_every ENC_UPDATE_EVERY \
  --mom_pred MOM_PRED \
  --mom_enc MOM_ENC \
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

### Analysis

```
python analysis/heatmaps.py \
  --runs_dir runs_parquet \
  --lr_enc_mode LR_ENC_MODE \
  --metric W_Fro_aligned \
  --epoch 2000 \
  --out figs \
  --use_last_valid \
  --enc_metric BA_err_aligned
```

## Hyperparameters

### Single Run

- **SEED** (int): Random seed used for numpy and torch. Default: 42
- **DEVICE** (string): Device to run on. Use cuda or cpu. Default: cuda
- **D_LATENT** (int): Latent dimension of the synthetic z. Default: 10
- **D_OBS** (int): Observed dimension of x. Mixing matrix A has shape D_OBS x D_LATENT. Default: 100
- **EPOCHS** (int): Number of training epochs per run. Default: 2000
- **LOG_EVERY** (int): Evaluation and logging period in epochs. Default: 10
- **BATCH_SIZE** (int): Minibatch size for training. Default: 64
- **LR_PRED** (float): Predictor learning rate (alpha). Controls how fast the predictor tracks the teacher features. Default: 1e-3
- **LR_ENC** (float): Encoder learning rate (beta) for single-run mode. Ignored if lr_enc_mode in sweep overrides per-run values. Default: 1e-3
- **MOM_PRED** (float): Momentum for the predictor optimizer if used. Typically not the EMA; see EMA_M for the teacher. Default: 0.95
- **MOM_ENC** (float): Momentum for the encoder optimizer if used. Default: 0.95
- **PRED_UPDATE_EVERY** (int): Update frequency for the predictor in steps. 1 means update every step. Default: 1
- **ENC_UPDATE_EVERY** (int): Update frequency for the online encoder in steps. 1 means update every step. Default: 1
- **EMA_M** (float): EMA momentum m for the target encoder. The EMA rate is tau = 1 - m. Default: 0.999
- **SIGMA_N** (float): Observation noise standard deviation in x = A z + epsilon. Default: 0.5
- **ENC** (string): Encoder architecture for both online and target branches. Default: linear (choices: ['linear', 'mlp'])
- **PRED** (string): Predictor architecture. Default: linear (choices: ['linear', 'mlp'])
- **LOG_WANDB** (bool): Enable logging to Weights and Biases when true. Default: False
- **AUTO_ALPHA_C** (float): Auto schedule constant c for alpha_t = c * tau / H_t. Set 0 to disable auto alpha. Default: 0.4
- **AUTO_BETA_C** (float): Auto schedule constant c for beta_t. If > 0, beta_t is set to c * tau / H_t as well. Set 0 to disable. Default: 0.4

### Sweep

- **SWEEP_SEEDS** (list[int]): List of seeds for repeated runs at each grid point. Default: [42, 43]
- **SWEEP_BATCHES** (list[int]): Batch sizes to sweep. Default: [32, 64]
- **LR_PRED_LOGSPACE** (list[number]): Predictor LR grid in logspace as [start, stop, steps]. Default: ['1e-5', '1e-2', 4]
- **TAU_LOGSPACE** (list[number]): EMA rate tau grid in logspace as [start, stop, steps]. Note tau = 1 - EMA_M. Default: ['1e-4', '1e-1', 4]
- **LR_ENC_MODE** (string): How to set encoder LR (beta) in the sweep:
  same: beta = alpha
  ratio: beta = ENC_OVER_PRED * alpha
  logspace: take beta grid from LR_ENC_LOGSPACE
  from_tau: beta = BETA_FROM_TAU_C * tau
  auto: beta is computed in-train from curvature like alpha (beta_t ~= c * tau / H_t) Default: auto (choices: ['same', 'ratio', 'logspace', 'from_tau', 'auto'])
- **ENC_OVER_PRED** (float): Used when lr_enc_mode = ratio. Sets beta = ENC_OVER_PRED * alpha. Default: 0.4
- **LR_ENC_LOGSPACE** (list[number]): Used when lr_enc_mode = logspace. Beta grid in logspace as [start, stop, steps]. Default: ['1e-5', '1e-2', 6]
- **BETA_FROM_TAU_C** (float): Used when lr_enc_mode = from_tau. Sets beta = BETA_FROM_TAU_C * tau. Default: 0.4
- **SWEEP_PRED_UPDATE_EVERY** (int): Predictor update frequency used during sweep runs. Default: 1
- **SWEEP_ENC_UPDATE_EVERY** (int): Encoder update frequency used during sweep runs. Default: 1
- **SWEEP_MOM_PRED** (float): Predictor optimizer momentum used during sweep runs. Default: 0.9
- **SWEEP_MOM_ENC** (float): Encoder optimizer momentum used during sweep runs. Default: 0.9

### Analysis

- **ANALYSIS_METRIC** (string): Predictor metric used for the main heatmap. W_Fro_aligned is log10 Frobenius error to analytic P* after Procrustes alignment. Default: W_Fro_aligned
- **ENC_ANALYSIS_METRIC** (string): Encoder metric used for the encoder heatmap. BA_err_aligned is R-aligned error between learned encoder and A+. Default: BA_err_aligned
- **ANALYSIS_EPOCH** (int): Reference epoch for selecting values or the cutoff for last-valid selection. Default: 2000
- **INSTALL_REQS** (bool): When true, the Windows batch script will pip install requirements into the active conda env. Default: False

## Auto Schedule

### Rationale

We adapt alpha_t using an online curvature estimate H_t = lambda_max(C_oo,t), the largest
eigenvalue of the online feature covariance. In the scalar noisy model (whitened), the
stationary variance of the student-teacher gap behaves like Var(e_inf) ~= (alpha^2 / (2*(alpha + tau))) * sigma_g^2.
For fixed tau, a useful regime is alpha <= tau: fast enough to track, not so large that noise dominates.
For curvature h != 1 the effective step is alpha*h, hence alpha*h ~= c * tau which gives
alpha_t = c * tau / h. Using the worst-case curvature yields a single global step:
alpha_t = c * tau / H_t. We set beta_t similarly when lr_enc_mode = auto.

### Estimation

H_t is estimated by a few steps of power iteration on the minibatch (or EMA) covariance of z_o.
A single iteration per logging step is typically sufficient.

## Metrics Logged

- **W_Fro_aligned**: Log10 Frobenius norm error between learned predictor P and analytic P* after Procrustes alignment (R-aligned).
- **W_rel_sv_err**: Relative singular value error of the predictor on top-r directions.
- **BA_err_aligned**: R-aligned error between learned encoder and the pseudoinverse A+.
- **enc_proj_dist**: Projection distance between spans of B_o^T and A+^T.
- **enc_procrustes_fro**: Procrustes-aligned Frobenius error for the encoder.
- **erank_pred**: Effective rank of P z_o to check for collapse in predictor outputs.
- **erank_encoded**: Effective rank of z_o to check for collapse in encoder outputs.
- **loss**: Training loss; useful for U-shaped loss vs alpha slices.
- **Hmax**: Curvature estimate H_t = lambda_max(C_oo,t) used for overlays and auto schedule.

## Outputs

### Parquet

Each run writes a Parquet file to runs_parquet with per-epoch metrics and hyperparameters.

### Figures

analysis/heatmaps.py writes PNGs to figs:
  heatmap_<metric>_bs<batch>_epoch<epoch>.png
  slices_loss_vs_alpha_bs<batch>_epoch<epoch>.png
  heatmap_erank_encoded_* and heatmap_erank_pred_* when available.

## Notes

1) EMA rate is tau = 1 - EMA_M. High momentum means small tau.
2) When lr_enc_mode = auto, both alpha_t and beta_t follow the curvature-scaled rule with constants AUTO_ALPHA_C and AUTO_BETA_C.
3) For linear encoders and predictor, analytic optima are available:
   P* = C_to C_oo^{-1} (Wiener shrinkage), and B_o converges toward A+ up to an orthogonal rotation (R-aligned metrics reflect this).
4) The figures and metrics are designed to support the conclusions about preferred regimes:
   predictor prefers alpha slightly above tau; encoder prefers higher momentum (small tau) and smaller beta; auto mode balances both.

## References

- [byol] Grill et al., Bootstrap Your Own Latent (BYOL), NeurIPS 2020.
- [jepa] Assran et al., The Joint Embedding Predictive Architecture (JEPA), 2023.
- [rdm] Representation Dynamics and Mutual Information (RDM), arXiv:2303.02387.
- [wiener] Wiener, Extrapolation, Interpolation, and Smoothing of Stationary Time Series, 1949.
- [matrix_cookbook] Matrix Cookbook identity vec(UXV) = (V^T ⊗ U) vec(X) used in the encoder update derivation.

---