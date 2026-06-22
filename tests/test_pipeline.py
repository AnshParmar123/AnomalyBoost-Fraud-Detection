import os
import pytest
import numpy as np
import pandas as pd
from src.pipeline import train_pipeline, predict_pipeline
from src.config import AUTOENCODER_MODEL_FILE, XGB_MODEL_FILE, SCALER_FILE, METADATA_FILE, SCORED_CSV_FILE

def test_end_to_end_pipeline(tmp_path):
    # 1. Create a dummy dataset
    np.random.seed(42)
    rows = 150
    data = {
        'Time': np.arange(rows) * 10,
        'Amount': np.random.exponential(scale=50.0, size=rows),
        'Class': [0] * (rows - 5) + [1] * 5  # Imbalanced label representation
    }
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(size=rows)
    df = pd.DataFrame(data)
    
    csv_input_path = os.path.join(tmp_path, "dummy_creditcard.csv")
    df.to_csv(csv_input_path, index=False)
    
    # 2. Run training pipeline
    model_dir = os.path.join(tmp_path, "models")
    config_override = {
        'autoencoder_epochs': 2,        # Fast epoch count for testing
        'autoencoder_batch_size': 32,
        'xgb_use_cv': False,             # Disable CV to speed up test execution
        'seed': 42
    }
    
    meta = train_pipeline(csv_input_path, model_dir, config_override)
    
    # Check outputs generated
    assert os.path.exists(os.path.join(model_dir, AUTOENCODER_MODEL_FILE))
    assert os.path.exists(os.path.join(model_dir, XGB_MODEL_FILE))
    assert os.path.exists(os.path.join(model_dir, SCALER_FILE))
    assert os.path.exists(os.path.join(model_dir, METADATA_FILE))
    assert os.path.exists(os.path.join(model_dir, SCORED_CSV_FILE))
    
    assert 'xgb_test_auc' in meta['metrics']
    assert 'recon_only_auc' in meta['metrics']
    
    # 3. Run prediction pipeline
    output_pred_csv = os.path.join(tmp_path, "predictions.csv")
    pred_df = predict_pipeline(csv_input_path, model_dir, output_pred_csv)
    
    assert os.path.exists(output_pred_csv)
    assert 'recon_error' in pred_df.columns
    assert 'xgb_prob' in pred_df.columns
    assert 'is_fraud_predicted' in pred_df.columns
    assert len(pred_df) == rows
