import os
import json
import logging
import datetime
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import DEFAULT_CONFIG, AUTOENCODER_MODEL_FILE, XGB_MODEL_FILE, SCALER_FILE, PCA_FILE, SHAP_SAMPLE_FILE, SCORED_CSV_FILE, METADATA_FILE, ARTIFACTS_JSON_FILE
from src.data import load_dataset, preprocess_data, prepare_autoencoder_data
from src.models import build_dense_autoencoder, build_vae
from src.evaluate import evaluate_autoencoder_reconstruction, evaluate_supervised_model, plot_training_loss
from src.explain import generate_shap_explanations, get_feature_importance

logger = logging.getLogger(__name__)

def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def train_pipeline(csv_path: str, output_dir: str, config_override: dict = None) -> dict:
    """Ties together data loading, preprocessing, model training, evaluation, and explainability."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration setup
    config = DEFAULT_CONFIG.copy()
    if config_override:
        config.update(config_override)
    
    set_seed(config['seed'])
    logger.info(f"Starting training pipeline with config: {config}")
    
    # 1. Load Data
    df = load_dataset(csv_path)
    
    # 2. Preprocess Data
    X_scaled, y, df_clean, scaler, pca = preprocess_data(df, config, fit_scaler=True)
    
    # Save preprocessing models
    joblib.dump(scaler, os.path.join(output_dir, SCALER_FILE))
    if pca:
        joblib.dump(pca, os.path.join(output_dir, PCA_FILE))
        
    # 3. Train Autoencoder
    X_train_ae, X_val_ae = prepare_autoencoder_data(X_scaled, y, seed=config['seed'])
    input_dim = X_train_ae.shape[1]
    encoding_dim = max(8, input_dim // 4)
    
    if config['use_vae']:
        model_ae = build_vae(input_dim, latent_dim=encoding_dim)
    else:
        model_ae = build_dense_autoencoder(input_dim, encoding_dim)
        
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    
    logger.info("Training Autoencoder...")
    if config['use_vae']:
        # VAE has loss inside add_loss and doesn't need Y targets
        history = model_ae.fit(
            X_train_ae, epochs=config['autoencoder_epochs'], 
            batch_size=config['autoencoder_batch_size'], 
            validation_data=(X_val_ae, None), 
            callbacks=[es], verbose=1
        )
    else:
        history = model_ae.fit(
            X_train_ae, X_train_ae, 
            epochs=config['autoencoder_epochs'], 
            batch_size=config['autoencoder_batch_size'], 
            validation_data=(X_val_ae, X_val_ae), 
            callbacks=[es], verbose=1
        )
        
    # Save Autoencoder Model
    model_ae.save(os.path.join(output_dir, AUTOENCODER_MODEL_FILE))
    plot_training_loss(history, output_dir)
    
    # 4. Generate Reconstruction Error Feature
    mse, ae_metrics = evaluate_autoencoder_reconstruction(model_ae, X_train_ae, X_scaled, y, output_dir)
    
    # 5. Feature Augmentation (Scale Features + Reconstruction Error)
    X_for_xgb = np.hstack([X_scaled, mse.reshape(-1, 1)])
    feature_names = [f"f_{i}" for i in range(X_scaled.shape[1])] + ['recon_error']
    if hasattr(df_clean, 'columns'):
        # Map original feature names
        orig_cols = list(df_clean.columns)
        for label_col in ['Class', 'isFraud', 'fraud', 'label', 'is_fraud']:
            if label_col in orig_cols:
                orig_cols.remove(label_col)
        # Handle time removal if occurred
        if 'Time' in df_clean.columns and 'Time' not in orig_cols:
            pass # already dropped
        feature_names = orig_cols + ['recon_error']
        
    # If no labels, stop here (unsupervised)
    if y is None:
        logger.info("No labels available. Supervised training skipped.")
        df_clean['recon_error'] = mse
        df_clean.to_csv(os.path.join(output_dir, SCORED_CSV_FILE), index=False)
        return {'recon_only_metrics': ae_metrics}
        
    # 6. Train XGBoost Supervised Model
    logger.info("Training XGBoost Classifier...")
    early_stop_rounds = config.get('early_stopping_rounds', 20)
    
    if config['xgb_use_cv']:
        skf = StratifiedKFold(n_splits=config['xgb_n_splits'], shuffle=True, random_state=config['seed'])
        aucs = []
        models_cv = []
        fold = 0
        
        for train_idx, val_idx in skf.split(X_for_xgb, y):
            fold += 1
            Xtr, Xv = X_for_xgb[train_idx], X_for_xgb[val_idx]
            ytr, yv = y.iloc[train_idx], y.iloc[val_idx]
            
            clf = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric='auc',
                random_state=config['seed']
            )
            
            pos = np.sum(ytr == 1)
            neg = np.sum(ytr == 0)
            if pos > 0:
                clf.set_params(scale_pos_weight=max(1, neg // (pos + 1)))
                
            try:
                clf.fit(Xtr, ytr, eval_set=[(Xv, yv)], early_stopping_rounds=early_stop_rounds, verbose=False)
            except TypeError:
                clf.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
                
            prob = clf.predict_proba(Xv)[:, 1]
            auc_fold = roc_auc_score(yv, prob)
            logger.info(f"Fold {fold} ROC AUC: {auc_fold:.4f}")
            aucs.append(auc_fold)
            models_cv.append(clf)
            
        best_idx = int(np.argmax(aucs))
        best_clf = models_cv[best_idx]
        logger.info(f"Selected fold {best_idx+1} model as best (CV AUC mean: {np.mean(aucs):.4f}, std: {np.std(aucs):.4f})")
    else:
        # Single split training
        X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
            X_for_xgb, y, test_size=0.2, stratify=y, random_state=config['seed']
        )
        best_clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='auc',
            random_state=config['seed']
        )
        pos = np.sum(y_train_xgb == 1)
        neg = np.sum(y_train_xgb == 0)
        best_clf.set_params(scale_pos_weight=max(1, neg // (pos + 1)))
        
        try:
            best_clf.fit(X_train_xgb, y_train_xgb, eval_set=[(X_val_xgb, y_val_xgb)], early_stopping_rounds=early_stop_rounds, verbose=False)
        except TypeError:
            best_clf.fit(X_train_xgb, y_train_xgb, eval_set=[(X_val_xgb, y_val_xgb)], verbose=False)
            
    # Save Best XGBoost Model
    joblib.dump(best_clf, os.path.join(output_dir, XGB_MODEL_FILE))
    
    # 7. Evaluate Supervised Model (on full dataset using predict_proba)
    xgb_probs = best_clf.predict_proba(X_for_xgb)[:, 1]
    xgb_metrics = evaluate_supervised_model(y, xgb_probs, output_dir)
    
    # 8. SHAP Explainability & Feature Importance
    generate_shap_explanations(best_clf, X_for_xgb, feature_names, output_dir)
    get_feature_importance(best_clf, feature_names, output_dir)
    
    # 9. Save predictions
    df_clean['recon_error'] = mse
    df_clean['xgb_prob'] = xgb_probs
    df_clean['is_fraud_predicted'] = (xgb_probs > 0.5).astype(int)
    scored_csv_path = os.path.join(output_dir, SCORED_CSV_FILE)
    df_clean.to_csv(scored_csv_path, index=False)
    logger.info(f"Saved transactions with scores to: {scored_csv_path}")
    
    # 10. Metadata and Artifact Registry
    meta = {
        'run_timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
        'config': config,
        'metrics': {**ae_metrics, **xgb_metrics}
    }
    with open(os.path.join(output_dir, METADATA_FILE), 'w') as f:
        json.dump(meta, f, indent=4)
        
    artifacts = {
        'autoencoder_model': AUTOENCODER_MODEL_FILE,
        'xgb_model': XGB_MODEL_FILE,
        'scaler': SCALER_FILE,
        'pca': PCA_FILE if config['use_pca'] else None,
        'shap_sample': SHAP_SAMPLE_FILE,
        'scored_csv': SCORED_CSV_FILE,
        'metadata': METADATA_FILE
    }
    with open(os.path.join(output_dir, ARTIFACTS_JSON_FILE), 'w') as f:
        json.dump(artifacts, f, indent=4)
        
    logger.info("Training pipeline run completed successfully.")
    return meta

def predict_pipeline(csv_path: str, model_dir: str, output_csv_path: str):
    """Loads pre-trained scaler, autoencoder, and classifier to compute anomaly scores and fraud probability."""
    logger.info(f"Starting inference pipeline. Reading models from: {model_dir}")
    
    # Load Preprocessing artifacts
    scaler = joblib.load(os.path.join(model_dir, SCALER_FILE))
    pca = None
    pca_path = os.path.join(model_dir, PCA_FILE)
    if os.path.exists(pca_path):
        pca = joblib.load(pca_path)
        
    # Read configs from metadata if exists
    config = DEFAULT_CONFIG
    metadata_path = os.path.join(model_dir, METADATA_FILE)
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            config = json.load(f).get('config', DEFAULT_CONFIG)
            
    # Load Model artifacts
    model_ae = tf.keras.models.load_model(os.path.join(model_dir, AUTOENCODER_MODEL_FILE), compile=False)
    best_clf = joblib.load(os.path.join(model_dir, XGB_MODEL_FILE))
    
    # Load Input Data
    df = load_dataset(csv_path)
    
    # Preprocess
    X_scaled, _, df_clean, _, _ = preprocess_data(df, config, fit_scaler=False, scaler=scaler, pca=pca)
    
    # AE Reconstruction
    X_pred = model_ae.predict(X_scaled)
    mse = np.mean(np.square(X_scaled - X_pred), axis=1)
    
    # Augment features
    X_for_xgb = np.hstack([X_scaled, mse.reshape(-1, 1)])
    
    # XGB predict
    xgb_probs = best_clf.predict_proba(X_for_xgb)[:, 1]
    
    # Save outputs
    df_clean['recon_error'] = mse
    df_clean['xgb_prob'] = xgb_probs
    df_clean['is_fraud_predicted'] = (xgb_probs > 0.5).astype(int)
    
    df_clean.to_csv(output_csv_path, index=False)
    logger.info(f"Saved prediction results to: {output_csv_path}")
    return df_clean
