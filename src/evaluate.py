import logging
import os
import matplotlib
# Use Agg backend for headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    confusion_matrix, 
    precision_recall_fscore_support, 
    classification_report, 
    roc_curve
)

logger = logging.getLogger(__name__)

def evaluate_autoencoder_reconstruction(model_ae, X_train, X_test, y_test, output_dir=None):
    """
    Computes reconstruction error (MSE) and evaluates AE reconstruction quality.
    
    Returns:
        mse_test: reconstruction errors for test set.
        metrics: dict of computed stats.
    """
    logger.info("Computing Autoencoder reconstruction errors...")
    X_test_pred = model_ae.predict(X_test)
    mse_test = np.mean(np.square(X_test - X_test_pred), axis=1)
    
    metrics = {}
    if y_test is not None:
        auc_score = roc_auc_score(y_test, mse_test)
        metrics['recon_only_auc'] = float(auc_score)
        logger.info(f"Autoencoder Reconstruction-only ROC AUC: {auc_score:.4f}")
        
        # Calculate threshold on training normals (mean + 3 * std)
        X_train_pred = model_ae.predict(X_train)
        mse_train = np.mean(np.square(X_train - X_train_pred), axis=1)
        threshold = float(np.mean(mse_train) + 3 * np.std(mse_train))
        metrics['threshold'] = threshold
        logger.info(f"Reconstruction error threshold (mean + 3*std on train): {threshold:.6f}")
        
        y_pred = (mse_test > threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        metrics['precision'] = float(precision)
        metrics['recall'] = float(recall)
        metrics['f1'] = float(f1)
        
        logger.info(f"Reconstruction-only Detector: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        # Save reconstruction error distributions
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.hist(mse_test[y_test == 0], bins=50, alpha=0.7, color='blue', label='Normal')
            plt.title('Reconstruction Error: Normal')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.hist(mse_test[y_test == 1], bins=50, alpha=0.7, color='red', label='Fraud')
            plt.title('Reconstruction Error: Fraud')
            plt.legend()
            
            fig_path = os.path.join(output_dir, 'ae_error_distribution.png')
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close()
            logger.info(f"Saved reconstruction error distribution plot to: {fig_path}")
            
    return mse_test, metrics

def evaluate_supervised_model(y_true, y_prob, output_dir=None):
    """
    Evaluates XGBoost model predictions.
    
    Returns:
        metrics: dict of classification report stats.
    """
    logger.info("Evaluating XGBoost model...")
    y_pred = (y_prob > 0.5).astype(int)
    auc_score = roc_auc_score(y_true, y_prob)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    metrics = {
        'xgb_test_auc': float(auc_score),
        'xgb_precision': float(precision),
        'xgb_recall': float(recall),
        'xgb_f1': float(f1)
    }
    
    logger.info(f"XGBoost Test ROC AUC: {auc_score:.4f}")
    logger.info("Classification Report:")
    report = classification_report(y_true, y_pred, digits=4)
    print(report)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('XGBoost Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        cm_path = os.path.join(output_dir, 'xgb_confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        logger.info(f"Saved confusion matrix to: {cm_path}")
        
        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'XGBoost (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        roc_path = os.path.join(output_dir, 'xgb_roc_curve.png')
        plt.savefig(roc_path)
        plt.close()
        logger.info(f"Saved ROC curve to: {roc_path}")
        
    return metrics

def plot_training_loss(history, output_dir):
    """Plots and saves the Autoencoder training/validation loss curve."""
    if not history or not hasattr(history, 'history'):
        return
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(6, 4))
    plt.plot(history.history.get('loss', []), label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Autoencoder Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    
    loss_path = os.path.join(output_dir, 'ae_training_loss.png')
    plt.savefig(loss_path)
    plt.close()
    logger.info(f"Saved loss curve plot to: {loss_path}")
