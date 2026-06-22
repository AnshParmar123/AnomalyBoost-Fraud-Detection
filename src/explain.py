import logging
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap

logger = logging.getLogger(__name__)

def generate_shap_explanations(best_clf, X_for_xgb, feature_names, output_dir, max_samples=2000):
    """
    Computes SHAP explanations on XGBoost classifier predictions.
    Saves SHAP sample values to CSV and plots summary.
    """
    logger.info("Computing SHAP values (TreeExplainer)...")
    try:
        explainer = shap.TreeExplainer(best_clf)
        
        # Sample data to make SHAP fast
        num_samples = min(max_samples, X_for_xgb.shape[0])
        sample_idx = np.random.choice(X_for_xgb.shape[0], size=num_samples, replace=False)
        sample = X_for_xgb[sample_idx]
        
        shap_values = explainer.shap_values(sample)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save sample SHAP values to CSV
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        shap_csv_path = os.path.join(output_dir, 'shap_values_sample.csv')
        shap_df.to_csv(shap_csv_path, index=False)
        logger.info(f"Saved SHAP values sample to: {shap_csv_path}")
        
        # Save SHAP Summary Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, pd.DataFrame(sample, columns=feature_names), show=False)
        shap_plot_path = os.path.join(output_dir, 'shap_summary_plot.png')
        plt.tight_layout()
        plt.savefig(shap_plot_path)
        plt.close()
        logger.info(f"Saved SHAP summary plot to: {shap_plot_path}")
        
        return shap_values, sample
    except Exception as e:
        logger.error(f"Error computing SHAP values: {e}")
        return None, None

def get_feature_importance(best_clf, feature_names, output_dir=None):
    """
    Extracts feature importance based on XGBoost gain metric.
    """
    logger.info("Extracting feature importances from XGBoost...")
    booster = best_clf.get_booster() if hasattr(best_clf, 'get_booster') else None
    
    if booster is None:
        logger.warning("Could not obtain booster from XGBoost classifier to compute feature importance.")
        return None
        
    importance_dict = booster.get_score(importance_type='gain')
    imp_items = []
    
    for k, v in importance_dict.items():
        # XGBoost feature keys might be f0, f1, etc. or actual names
        try:
            if k.startswith('f'):
                idx = int(k[1:])
                fname = feature_names[idx] if idx < len(feature_names) else k
            else:
                fname = k
        except Exception:
            fname = k
        imp_items.append((fname, v))
        
    if not imp_items:
        logger.warning("Feature importance dictionary is empty.")
        return None
        
    imp_df = pd.DataFrame(imp_items, columns=['feature', 'gain']).sort_values('gain', ascending=False)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Save top features plot
        plt.figure(figsize=(8, 6))
        sns.barplot(x='gain', y='feature', data=imp_df.head(30))
        plt.title('Top 30 XGBoost Features (by Gain)')
        fig_path = os.path.join(output_dir, 'xgb_feature_importance.png')
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        logger.info(f"Saved feature importance plot to: {fig_path}")
        
        # Save importance csv
        imp_csv_path = os.path.join(output_dir, 'feature_importance.csv')
        imp_df.to_csv(imp_csv_path, index=False)
        
    return imp_df
