import os
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def load_dataset(csv_path: str) -> pd.DataFrame:
    """Loads the CSV transaction dataset."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV dataset not found at: {csv_path}")
    logger.info(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded dataset shape: {df.shape}")
    return df

def preprocess_data(df: pd.DataFrame, config: dict, fit_scaler: bool = True, 
                    scaler=None, pca=None):
    """
    Cleans, encodes, scales, and runs PCA on features.
    
    Args:
        df: Input DataFrame.
        config: Configurations dict.
        fit_scaler: If True, fits and saves the scaler and pca.
        scaler: Pre-fitted scaler if fit_scaler is False.
        pca: Pre-fitted PCA if fit_scaler is False.
        
    Returns:
        X_scaled: Transformed features as a NumPy array.
        y: Target label (Series or None).
        df_clean: Cleaned original dataframe copy.
        scaler: The fitted/used scaler.
        pca: The fitted/used PCA model.
    """
    df_clean = df.copy()
    
    # Detect label column
    label_col = None
    for c in ['Class', 'isFraud', 'fraud', 'label', 'is_fraud']:
        if c in df_clean.columns:
            label_col = c
            break
            
    if label_col:
        logger.info(f"Detected label column: {label_col}")
        y = df_clean[label_col].astype(int).copy()
        X_df = df_clean.drop(columns=[label_col])
    else:
        logger.info("No label column detected. Proceeding in unsupervised mode.")
        y = None
        X_df = df_clean.copy()
        
    # Time feature engineering
    if 'Time' in X_df.columns:
        try:
            X_df['hour'] = (X_df['Time'] // 3600) % 24
            logger.info("Engineered 'hour' feature from 'Time'.")
        except Exception as e:
            logger.warning(f"Failed to engineer 'hour' feature: {e}")
        X_df = X_df.drop(columns=['Time'])
        
    # One-hot encode low-cardinality categoricals
    cat_cols = [c for c in X_df.columns if X_df[c].dtype == 'object' or X_df[c].nunique() < 50]
    cat_cols = [c for c in cat_cols if X_df[c].nunique() < 200]
    if len(cat_cols) > 0:
        logger.info(f"One-hot encoding categorical features: {cat_cols}")
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)
        
    # Impute missing values with median
    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        if X_df[c].isnull().any():
            median_val = X_df[c].median()
            X_df[c] = X_df[c].fillna(median_val)
            
    # Clip outliers to 1st and 99th percentile to stabilize models
    for c in num_cols:
        low, high = X_df[c].quantile(0.01), X_df[c].quantile(0.99)
        # Avoid clipping if low and high are identical
        if low < high:
            X_df[c] = X_df[c].clip(lower=low, upper=high)
            
    # Scale features
    X_vals = X_df.values
    if fit_scaler:
        if config.get('use_robust_scaler', True):
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_vals)
    else:
        if scaler is None:
            raise ValueError("Pre-fitted scaler must be supplied when fit_scaler is False.")
        X_scaled = scaler.transform(X_vals)
        
    # Optional PCA dimensionality reduction
    if config.get('use_pca', False):
        if fit_scaler:
            pca = PCA(n_components=config.get('pca_n_components', 20), random_state=config.get('seed', 42))
            X_scaled = pca.fit_transform(X_scaled)
        else:
            if pca is None:
                raise ValueError("Pre-fitted PCA must be supplied when fit_scaler is False and use_pca is True.")
            X_scaled = pca.transform(X_scaled)
            
    return X_scaled, y, df_clean, scaler, pca

def prepare_autoencoder_data(X_scaled: np.ndarray, y: pd.Series, seed: int = 42):
    """
    Prepares training and validation datasets for the Autoencoder.
    Under semi-supervised learning, the Autoencoder trains on normal instances only.
    """
    if y is not None:
        normal_idx = np.where(y == 0)[0]
        X_train_full = X_scaled[normal_idx]
        X_train, X_val = train_test_split(X_train_full, test_size=0.2, random_state=seed)
        logger.info(f"Semi-supervised split: training Autoencoder on {X_train.shape[0]} normal samples (Val: {X_val.shape[0]})")
    else:
        X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=seed)
        logger.info(f"Unsupervised split: training Autoencoder on {X_train.shape[0]} samples (Val: {X_val.shape[0]})")
        
    return X_train, X_val
