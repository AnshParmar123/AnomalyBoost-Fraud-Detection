# AnomalyBoost Fraud Detection

AnomalyBoost is a hybrid credit card fraud detection project that combines an autoencoder-based anomaly signal with an XGBoost classifier. The unsupervised model learns the reconstruction pattern of mostly normal transactions, and its reconstruction error becomes an additional feature for the supervised fraud classifier.

## What is included

- `src/`: training, preprocessing, explainability, evaluation, and CLI entrypoints
- `app.py`: Streamlit dashboard for model metrics, inference, and visual inspection
- `tests/`: lightweight regression tests for data utilities, models, and pipeline flow
- `ccfd.ipynb`: exploratory notebook
- `requirements.txt`: Python dependencies

Large generated files such as trained models, scored CSV outputs, SHAP exports, plots, and other artifacts are intentionally ignored so the repository stays lightweight.

## Pipeline overview

1. Load and preprocess transaction data
2. Train an autoencoder on normal transactions
3. Compute reconstruction error for each transaction
4. Append reconstruction error as a feature
5. Train an XGBoost classifier on the augmented feature set
6. Export metrics, plots, and optional explainability artifacts

## Quickstart

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the pipeline

```bash
python -m src.cli train --csv-path /path/to/creditcard.csv --output-dir outputs
```

Optional flags:

- `--vae`: use the variational autoencoder path
- `--pca`: enable PCA before model training
- `--epochs 25`: override autoencoder epochs
- `--batch-size 128`: override autoencoder batch size
- `--no-cv`: disable XGBoost cross-validation for faster runs

### 3. Score new transactions

```bash
python -m src.cli predict \
  --csv-path /path/to/new_transactions.csv \
  --model-dir outputs \
  --output-path outputs/predictions.csv
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

The dashboard expects trained artifacts to exist in the working directory or the directory you point it at during inference.

## Testing

```bash
pytest
```

## Notes

- The repository is set up to keep bulky artifacts out of Git history.
- The Streamlit app is designed for local experimentation rather than production deployment.
- The project works best with transaction datasets that resemble the common credit-card-fraud benchmark shape, including `Time`, `Amount`, and an optional label column such as `Class`.
