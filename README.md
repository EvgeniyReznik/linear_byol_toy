\# Linear BYOL Toy — Predictor as LPF, Encoder as Pseudoinverse



This project implements a neat, well-instrumented toy for BYOL-style learning on linear–Gaussian synthetic data:

\- \*\*Predictor\*\* learns the Wiener/low-pass shrinkage.

\- \*\*Online encoder\*\* is learned; \*\*target encoder\*\* is EMA of online.

\- You can use \*\*linear\*\* or \*\*MLP\*\* (nonlinear) encoders/predictors.



We log theory-driven metrics (Frobenius/SVD errors vs analytic optimum, subspace angles vs pseudoinverse, Procrustes error, effective ranks, curvature estimate) to \*\*Parquet\*\*; Weights \& Biases is optional.



\## Install



```bash

python -m venv .venv \&\& source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate

pip install -r requirements.txt



