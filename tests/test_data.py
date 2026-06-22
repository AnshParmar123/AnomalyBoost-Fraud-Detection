import pytest
import numpy as np
import pandas as pd
from src.data import preprocess_data, prepare_autoencoder_data
from src.config import DEFAULT_CONFIG

@pytest.fixture
def dummy_dataframe():
    np.random.seed(42)
    rows = 100
    data = {
        'Time': np.zeros(rows),  # Constant time resulting in constant hour = 0
        'Amount': np.random.exponential(scale=100.0, size=rows),
        'Class': np.random.choice([0, 1], size=rows, p=[0.95, 0.05])
    }
    # Add V1 to V28
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(size=rows)
    return pd.DataFrame(data)

def test_preprocess_data_fit(dummy_dataframe):
    config = DEFAULT_CONFIG.copy()
    config['use_pca'] = False
    
    X_scaled, y, df_clean, scaler, pca = preprocess_data(dummy_dataframe, config, fit_scaler=True)
    
    # Label class is removed from features (30 columns original - Class).
    # Time is dropped, hour is created but one-hot encoded as a constant column and dropped by drop_first=True.
    # Total remaining features: 28 (V1-V28) + 1 (Amount) = 29.
    assert X_scaled.shape[1] == 29  
    assert len(y) == 100
    assert scaler is not None
    assert pca is None

def test_preprocess_data_pca(dummy_dataframe):
    config = DEFAULT_CONFIG.copy()
    config['use_pca'] = True
    config['pca_n_components'] = 10
    
    X_scaled, y, df_clean, scaler, pca = preprocess_data(dummy_dataframe, config, fit_scaler=True)
    
    assert X_scaled.shape[1] == 10
    assert pca is not None

def test_prepare_autoencoder_data(dummy_dataframe):
    config = DEFAULT_CONFIG.copy()
    X_scaled, y, _, _, _ = preprocess_data(dummy_dataframe, config, fit_scaler=True)
    
    X_train, X_val = prepare_autoencoder_data(X_scaled, y, seed=42)
    
    # AE is trained only on normal data (Class = 0)
    normal_count = np.sum(y == 0)
    assert X_train.shape[0] + X_val.shape[0] == normal_count
