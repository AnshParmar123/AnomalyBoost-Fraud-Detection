import os

# Hyperparameters and Pipeline Settings
DEFAULT_CONFIG = {
    'use_robust_scaler': True,     # True for RobustScaler, False for StandardScaler
    'use_pca': False,              # Whether to apply PCA on scaled features
    'pca_n_components': 20,        # Number of components for PCA
    'use_vae': False,              # True to use Variational Autoencoder, False for Dense Autoencoder
    'autoencoder_epochs': 100,
    'autoencoder_batch_size': 256,
    'xgb_use_cv': True,            # Whether to train XGBoost with Cross-Validation
    'xgb_n_splits': 5,             # Folds for Cross-Validation
    'early_stopping_rounds': 20,   # XGBoost early stopping
    'seed': 42,
}

# Directories and Paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# Artifact Filenames
AUTOENCODER_MODEL_FILE = 'autoencoder_model.h5'
XGB_MODEL_FILE = 'xgb_best.joblib'
SCALER_FILE = 'scaler.joblib'
PCA_FILE = 'pca.joblib'
SHAP_SAMPLE_FILE = 'shap_values_sample.csv'
SCORED_CSV_FILE = 'transactions_with_scores_improved.csv'
METADATA_FILE = 'metadata.json'
ARTIFACTS_JSON_FILE = 'artifacts.json'
