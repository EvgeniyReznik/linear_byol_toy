@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ======================================================
REM  CONFIG — edit to taste
REM ======================================================
set MODE=%1
if "%MODE%"=="" set MODE=single

REM Conda env to use
set ENV_NAME=cuda_test

REM Output dirs
set OUT_DIR=runs_parquet
set FIG_DIR=figs

REM ---------- Single run defaults ----------
set SEED=42
set DEVICE=cuda
set D_LATENT=10
set D_OBS=100
set EPOCHS=2000
set LOG_EVERY=10
set BATCH_SIZE=64

REM separated time-scales
set LR_PRED=1e-3
set LR_ENC=1e-3
set MOM_PRED=0.95
set MOM_ENC=0.95
set PRED_UPDATE_EVERY=1
set ENC_UPDATE_EVERY=1

REM EMA teacher
set EMA_M=0.999
set SIGMA_N=0.5
set ENC=linear
set PRED=linear
set LOG_WANDB=false

REM ---------- Sweep settings (MODE=sweep) ----------
set SWEEP_SEEDS=42 43
set SWEEP_BATCHES=32 64

REM α (predictor) grid & τ grid (logspace: start stop steps)
set LR_PRED_LOGSPACE=1e-5 1e-2 4
set TAU_LOGSPACE=1e-4 1e-1 4

REM Choose how to set β (encoder LR): same | ratio | logspace | from_tau | auto
set LR_ENC_MODE=auto

REM if LR_ENC_MODE=ratio -> beta = ENC_OVER_PRED * alpha
set ENC_OVER_PRED=0.4

REM if LR_ENC_MODE=logspace -> take beta grid from this
set LR_ENC_LOGSPACE=1e-5 1e-2 6

REM if LR_ENC_MODE=from_tau -> beta = BETA_FROM_TAU_C * tau
set BETA_FROM_TAU_C=0.4

REM Optional automatic schedules inside train (α≈c·2/H, β≈c·τ/H). 0 disables.
set AUTO_ALPHA_C=0.4
set AUTO_BETA_C=0.4

REM pass-through for time-scale separation in sweep too
set SWEEP_MOM_PRED=0.9
set SWEEP_MOM_ENC=0.9
set SWEEP_PRED_UPDATE_EVERY=1
set SWEEP_ENC_UPDATE_EVERY=1

REM ---------- Analysis ----------
set ANALYSIS_METRIC=W_Fro_aligned
set ANALYSIS_EPOCH=2000
SET ENC_ANALYSIS_METRIC=BA_err_aligned

REM Whether to pip install project requirements inside the env (true/false)
set INSTALL_REQS=false

REM ======================================================
REM  Conda activation
REM ======================================================
where conda >NUL 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo [ERROR] 'conda' not found on PATH. Open "Anaconda Prompt" or add conda to PATH.
  exit /b 1
)

echo [conda] Activating environment "%ENV_NAME%" ...
call conda activate %ENV_NAME%
if %ERRORLEVEL% NEQ 0 (
  echo [ERROR] Failed to 'conda activate %ENV_NAME%'.
  echo        Make sure the env exists:   conda create -n %ENV_NAME% python=3.10
  exit /b 1
)

REM We are now inside the conda env; set python
set "PYEXE=python"

REM Ensure the src/ package is importable
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"

REM ======================================================
REM  Install project requirements (optional)
REM ======================================================
if /I "%INSTALL_REQS%"=="true" (
  if exist "requirements.txt" (
    echo [pip] Installing requirements into "%ENV_NAME%" ...
    %PYEXE% -m pip install --upgrade pip
    %PYEXE% -m pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
      echo [ERROR] pip install failed.
      exit /b 1
    )
  ) else (
    echo [warn] requirements.txt not found; skipping install.
  )
)

REM Create output folders
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
if not exist "%FIG_DIR%" mkdir "%FIG_DIR%"

REM ======================================================
REM  MODE: single | sweep | analyze | all
REM ======================================================
if /I "%MODE%"=="single"  goto :RUN_SINGLE
if /I "%MODE%"=="sweep"   goto :RUN_SWEEP
if /I "%MODE%"=="analyze" goto :RUN_ANALYSIS
if /I "%MODE%"=="all"     goto :RUN_ALL

echo [usage] run_conda_windows.bat ^<single^|sweep^|analyze^|all^>
exit /b 0


:RUN_SINGLE
echo [run] Single experiment in env "%ENV_NAME%" ...
set "RUN_NAME=seed%SEED%_bs%BATCH_SIZE%_alp%LR_PRED%_bet%LR_ENC%_m%EMA_M%"
%PYEXE% -m linear_ssl.train ^
  --seed %SEED% ^
  --device %DEVICE% ^
  --d_latent %D_LATENT% ^
  --d_obs %D_OBS% ^
  --epochs %EPOCHS% ^
  --log_every %LOG_EVERY% ^
  --batch_size %BATCH_SIZE% ^
  --lr_pred %LR_PRED% ^
  --lr_enc %LR_ENC% ^
  --mom_pred %MOM_PRED% ^
  --mom_enc %MOM_ENC% ^
  --pred_update_every %PRED_UPDATE_EVERY% ^
  --enc_update_every %ENC_UPDATE_EVERY% ^
  --ema_m %EMA_M% ^
  --sigma_n %SIGMA_N% ^
  --out_dir "%OUT_DIR%" ^
  --run_name "%RUN_NAME%" ^
  --enc %ENC% ^
  --pred %PRED% ^
  --auto_alpha_c %AUTO_ALPHA_C% ^
  --auto_beta_c %AUTO_BETA_C% ^
  --log_wandb %LOG_WANDB%
if %ERRORLEVEL% NEQ 0 (
  echo [ERROR] training failed.
  exit /b 1
)
echo [ok] Single run finished.
goto :EOF


:RUN_SWEEP
echo [run] Grid sweep (α × τ × batch) in env "%ENV_NAME%" ...
%PYEXE% scripts\run_sweep.py ^
  --out_dir "%OUT_DIR%" ^
  --seeds %SWEEP_SEEDS% ^
  --batch_sizes %SWEEP_BATCHES% ^
  --lr_pred_logspace %LR_PRED_LOGSPACE% ^
  --tau_logspace %TAU_LOGSPACE% ^
  --lr_enc_mode %LR_ENC_MODE% ^
  --enc_over_pred %ENC_OVER_PRED% ^
  --lr_enc_logspace %LR_ENC_LOGSPACE% ^
  --beta_from_tau_c %BETA_FROM_TAU_C% ^
  --pred_update_every %SWEEP_PRED_UPDATE_EVERY% ^
  --enc_update_every %SWEEP_ENC_UPDATE_EVERY% ^
  --mom_pred %SWEEP_MOM_PRED% ^
  --mom_enc %SWEEP_MOM_ENC% ^
  --auto_alpha_c %AUTO_ALPHA_C% ^
  --auto_beta_c %AUTO_BETA_C% ^
  --epochs %EPOCHS% ^
  --log_every %LOG_EVERY% ^
  --sigma_n %SIGMA_N% ^
  --d_latent %D_LATENT% ^
  --d_obs %D_OBS% ^
  --device %DEVICE% ^
  --enc %ENC% ^
  --pred %PRED% ^
  --log_wandb %LOG_WANDB%
if %ERRORLEVEL% NEQ 0 (
  echo [ERROR] sweep failed.
  exit /b 1
)
echo [ok] Sweep finished.
goto :EOF


:RUN_ANALYSIS
echo [analysis] Heatmaps and scatter (best-over-time) ...
%PYEXE% analysis\heatmaps.py ^
  --runs_dir "%OUT_DIR%" ^
  --lr_enc_mode %LR_ENC_MODE% ^
  --metric %ANALYSIS_METRIC% ^
  --epoch %ANALYSIS_EPOCH% ^
  --out "%FIG_DIR%" ^
  --use_last_valid ^
  --enc_metric %ENC_ANALYSIS_METRIC%
if %ERRORLEVEL% NEQ 0 (
  echo [ERROR] analysis failed.
  exit /b 1
)
echo [ok] Analysis finished → "%FIG_DIR%"
goto :EOF


:RUN_ALL
call :RUN_SINGLE
call :RUN_SWEEP
call :RUN_ANALYSIS
echo [ok] All steps complete.
goto :EOF
