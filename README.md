name: Linear BYOL Toy
summary: |-
  A small BYOL-style learning toy on linear Gaussian synthetic data.
  The predictor learns a Wiener (low-pass) shrinkage, the online encoder is learned,
  and the target encoder is an EMA of the online encoder. Encoders and predictor can
  be linear or small MLPs. Runs log theory-driven metrics to Parquet; Weights & Biases
  is optional.

project_structure:
  code_dirs:
    - src/
    - analysis/
    - scripts/
  entry_points:
    train: linear_ssl.train
    sweep: scripts/run_sweep.py
    analysis: analysis/heatmaps.py
  output_dirs:
    parquet_runs: runs_parquet
    figures: figs

install: |-
  # Recommended: use a clean Python or conda environment
  python -m venv .venv
  # On Unix/macOS
  source .venv/bin/activate
  # On Windows PowerShell
  .venv\Scripts\Activate.ps1

  pip install -r requirements.txt

quick_start: |-
  Windows convenience script (inside an activated conda env):
    run_conda_windows.bat <single|sweep|analyze|all>

  Examples:
    run_conda_windows.bat single
    run_conda_windows.bat sweep
    run_conda_windows.bat analyze
    run_conda_windows.bat all

modes:
  - name: single
    description: Run one training job with fixed hyperparameters. Writes a Parquet file per run.
  - name: sweep
    description: Grid over alpha (predictor LR), tau (EMA rate), batch size, and optional encoder LR scheme. Multiple seeds supported.
  - name: analyze
    description: Produce heatmaps and helper figures from runs_parquet into figs.
  - name: all
    description: Run single, then sweep, then analyze sequentially.

commands:
  training: |-
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
  sweep: |-
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
  analysis: |-
    python analysis/heatmaps.py \
      --runs_dir runs_parquet \
      --lr_enc_mode LR_ENC_MODE \
      --metric W_Fro_aligned \
      --epoch 2000 \
      --out figs \
      --use_last_valid \
      --enc_metric BA_err_aligned

hyperparameters:
  single_run:
    - key: SEED
      type: int
      default: 42
      description: Random seed used for numpy and torch.
    - key: DEVICE
      type: string
      default: cuda
      description: Device to run on. Use cuda or cpu.
    - key: D_LATENT
      type: int
      default: 10
      description: Latent dimension of the synthetic z.
    - key: D_OBS
      type: int
      default: 100
      description: Observed dimension of x. Mixing matrix A has shape D_OBS x D_LATENT.
    - key: EPOCHS
      type: int
      default: 2000
      description: Number of training epochs per run.
    - key: LOG_EVERY
      type: int
      default: 10
      description: Evaluation and logging period in epochs.
    - key: BATCH_SIZE
      type: int
      default: 64
      description: Minibatch size for training.
    - key: LR_PRED
      type: float
      default: 1e-3
      description: Predictor learning rate (alpha). Controls how fast the predictor tracks the teacher features.
    - key: LR_ENC
      type: float
      default: 1e-3
      description: Encoder learning rate (beta) for single-run mode. Ignored if lr_enc_mode in sweep overrides per-run values.
    - key: MOM_PRED
      type: float
      default: 0.95
      description: Momentum for the predictor optimizer if used. Typically not the EMA; see EMA_M for the teacher.
    - key: MOM_ENC
      type: float
      default: 0.95
      description: Momentum for the encoder optimizer if used.
    - key: PRED_UPDATE_EVERY
      type: int
      default: 1
      description: Update frequency for the predictor in steps. 1 means update every step.
    - key: ENC_UPDATE_EVERY
      type: int
      default: 1
      description: Update frequency for the online encoder in steps. 1 means update every step.
    - key: EMA_M
      type: float
      default: 0.999
      description: EMA momentum m for the target encoder. The EMA rate is tau = 1 - m.
    - key: SIGMA_N
      type: float
      default: 0.5
      description: Observation noise standard deviation in x = A z + epsilon.
    - key: ENC
      type: string
      default: linear
      choices: [linear, mlp]
      description: Encoder architecture for both online and target branches.
    - key: PRED
      type: string
      default: linear
      choices: [linear, mlp]
      description: Predictor architecture.
    - key: LOG_WANDB
      type: bool
      default: false
      description: Enable logging to Weights and Biases when true.
    - key: AUTO_ALPHA_C
      type: float
      default: 0.4
      description: Auto schedule constant c for alpha_t = c * tau / H_t. Set 0 to disable auto alpha.
    - key: AUTO_BETA_C
      type: float
      default: 0.4
      description: Auto schedule constant c for beta_t. If > 0, beta_t is set to c * tau / H_t as well. Set 0 to disable.

  sweep:
    - key: SWEEP_SEEDS
      type: list[int]
      default: [42, 43]
      description: List of seeds for repeated runs at each grid point.
    - key: SWEEP_BATCHES
      type: list[int]
      default: [32, 64]
      description: Batch sizes to sweep.
    - key: LR_PRED_LOGSPACE
      type: list[number]
      default: [1e-5, 1e-2, 4]
      description: Predictor LR grid in logspace as [start, stop, steps].
    - key: TAU_LOGSPACE
      type: list[number]
      default: [1e-4, 1e-1, 4]
      description: EMA rate tau grid in logspace as [start, stop, steps]. Note tau = 1 - EMA_M.
    - key: LR_ENC_MODE
      type: string
      default: auto
      choices: [same, ratio, logspace, from_tau, auto]
      description: |-
        How to set encoder LR (beta) in the sweep:
          same: beta = alpha
          ratio: beta = ENC_OVER_PRED * alpha
          logspace: take beta grid from LR_ENC_LOGSPACE
          from_tau: beta = BETA_FROM_TAU_C * tau
          auto: beta is computed in-train from curvature like alpha (beta_t ~= c * tau / H_t)
    - key: ENC_OVER_PRED
      type: float
      default: 0.4
      description: Used when lr_enc_mode = ratio. Sets beta = ENC_OVER_PRED * alpha.
    - key: LR_ENC_LOGSPACE
      type: list[number]
      default: [1e-5, 1e-2, 6]
      description: Used when lr_enc_mode = logspace. Beta grid in logspace as [start, stop, steps].
    - key: BETA_FROM_TAU_C
      type: float
      default: 0.4
      description: Used when lr_enc_mode = from_tau. Sets beta = BETA_FROM_TAU_C * tau.
    - key: SWEEP_PRED_UPDATE_EVERY
      type: int
      default: 1
      description: Predictor update frequency used during sweep runs.
    - key: SWEEP_ENC_UPDATE_EVERY
      type: int
      default: 1
      description: Encoder update frequency used during sweep runs.
    - key: SWEEP_MOM_PRED
      type: float
      default: 0.9
      description: Predictor optimizer momentum used during sweep runs.
    - key: SWEEP_MOM_ENC
      type: float
      default: 0.9
      description: Encoder optimizer momentum used during sweep runs.

  analysis:
    - key: ANALYSIS_METRIC
      type: string
      default: W_Fro_aligned
      description: Predictor metric used for the main heatmap. W_Fro_aligned is log10 Frobenius error to analytic P* after Procrustes alignment.
    - key: ENC_ANALYSIS_METRIC
      type: string
      default: BA_err_aligned
      description: Encoder metric used for the encoder heatmap. BA_err_aligned is R-aligned error between learned encoder and A+.
    - key: ANALYSIS_EPOCH
      type: int
      default: 2000
      description: Reference epoch for selecting values or the cutoff for last-valid selection.
    - key: INSTALL_REQS
      type: bool
      default: false
      description: When true, the Windows batch script will pip install requirements into the active conda env.

auto_schedule:
  rationale: |-
    We adapt alpha_t using an online curvature estimate H_t = lambda_max(C_oo,t), the largest
    eigenvalue of the online feature covariance. In the scalar noisy model (whitened), the
    stationary variance of the student-teacher gap behaves like Var(e_inf) ~= (alpha^2 / (2*(alpha + tau))) * sigma_g^2.
    For fixed tau, a useful regime is alpha <= tau: fast enough to track, not so large that noise dominates.
    For curvature h != 1 the effective step is alpha*h, hence alpha*h ~= c * tau which gives
    alpha_t = c * tau / h. Using the worst-case curvature yields a single global step:
    alpha_t = c * tau / H_t. We set beta_t similarly when lr_enc_mode = auto.
  estimation: |-
    H_t is estimated by a few steps of power iteration on the minibatch (or EMA) covariance of z_o.
    A single iteration per logging step is typically sufficient.

metrics_logged:
  - key: W_Fro_aligned
    description: Log10 Frobenius norm error between learned predictor P and analytic P* after Procrustes alignment (R-aligned).
  - key: W_rel_sv_err
    description: Relative singular value error of the predictor on top-r directions.
  - key: BA_err_aligned
    description: R-aligned error between learned encoder and the pseudoinverse A+.
  - key: enc_proj_dist
    description: Projection distance between spans of B_o^T and A+^T.
  - key: enc_procrustes_fro
    description: Procrustes-aligned Frobenius error for the encoder.
  - key: erank_pred
    description: Effective rank of P z_o to check for collapse in predictor outputs.
  - key: erank_encoded
    description: Effective rank of z_o to check for collapse in encoder outputs.
  - key: loss
    description: Training loss; useful for U-shaped loss vs alpha slices.
  - key: Hmax
    description: Curvature estimate H_t = lambda_max(C_oo,t) used for overlays and auto schedule.

outputs:
  parquet: |-
    Each run writes a Parquet file to runs_parquet with per-epoch metrics and hyperparameters.
  figures: |-
    analysis/heatmaps.py writes PNGs to figs:
      heatmap_<metric>_bs<batch>_epoch<epoch>.png
      slices_loss_vs_alpha_bs<batch>_epoch<epoch>.png
      heatmap_erank_encoded_* and heatmap_erank_pred_* when available.

notes: |-
  1) EMA rate is tau = 1 - EMA_M. High momentum means small tau.
  2) When lr_enc_mode = auto, both alpha_t and beta_t follow the curvature-scaled rule with constants AUTO_ALPHA_C and AUTO_BETA_C.
  3) For linear encoders and predictor, analytic optima are available:
     P* = C_to C_oo^{-1} (Wiener shrinkage), and B_o converges toward A+ up to an orthogonal rotation (R-aligned metrics reflect this).
  4) The figures and metrics are designed to support the conclusions about preferred regimes:
     predictor prefers alpha slightly above tau; encoder prefers higher momentum (small tau) and smaller beta; auto mode balances both.

references:
  - id: byol
    text: "Grill et al., Bootstrap Your Own Latent (BYOL), NeurIPS 2020."
  - id: jepa
    text: "Assran et al., The Joint Embedding Predictive Architecture (JEPA), 2023."
  - id: rdm
    text: "Representation Dynamics and Mutual Information (RDM), arXiv:2303.02387."
  - id: wiener
    text: "Wiener, Extrapolation, Interpolation, and Smoothing of Stationary Time Series, 1949."
  - id: matrix_cookbook
    text: "Matrix Cookbook identity vec(UXV) = (V^T ⊗ U) vec(X) used in the encoder update derivation."

