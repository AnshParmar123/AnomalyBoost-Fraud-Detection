import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="AnomalyBoost - Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 1.5rem;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1F2937;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allowed_html=True)

# Sidebar Information
st.sidebar.image("https://img.icons8.com/external-flatart-icons-outline-flatarticons/128/external-security-cyber-security-flatart-icons-outline-flatarticons-3.png", width=80)
st.sidebar.markdown("# AnomalyBoost")
st.sidebar.markdown("### Hybrid Fraud Detection")
st.sidebar.markdown(
    "A hybrid model coupling unsupervised reconstruction representation (Autoencoder/VAE) "
    "with supervised boosted classification (XGBoost) for state-of-the-art fraud detection."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Pipeline Controls")

# Header
st.markdown('<div class="main-title">AnomalyBoost Dashboard</div>', unsafe_allowed_html=True)
st.markdown('<div class="subtitle">Interactive Credit Card Fraud Detection Pipeline & Visualizations</div>', unsafe_allowed_html=True)

# Try loading metadata.json
metadata = {}
if os.path.exists("metadata.json"):
    try:
        with open("metadata.json", "r") as f:
            metadata = json.load(f)
    except Exception:
        pass

# Tabs
tab_overview, tab_inference, tab_visuals = st.tabs(["📊 Model Overview & Metrics", "🔍 Run Inference", "📈 Explainability & Visuals"])

with tab_overview:
    st.markdown("### Hybrid Architecture Workflow")
    st.markdown(
        "1. **Unsupervised Feature Representation:** A deep Autoencoder (Dense or VAE) is trained exclusively on normal transactions. "
        "It learns to reconstruct normal transaction patterns. Any fraud/anomalous transaction will suffer a high reconstruction error (MSE).\n"
        "2. **Feature Augmentation:** The reconstruction error is added as a new engineered feature to the original feature vector.\n"
        "3. **Supervised Classification:** An XGBoost classifier is trained on this augmented feature space to optimize detection precision and recall."
    )
    
    st.markdown("---")
    
    if metadata and 'metrics' in metadata:
        st.markdown("### Latest Training Runs Performance Metrics")
        metrics = metadata['metrics']
        
        # Display Metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">XGBoost Test AUC</div>'
                f'<div class="metric-value">{metrics.get("xgb_test_auc", 0.0):.4f}</div>'
                f'</div>', 
                unsafe_allowed_html=True
            )
        with col2:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Reconstruction AUC</div>'
                f'<div class="metric-value">{metrics.get("recon_only_auc", 0.0):.4f}</div>'
                f'</div>', 
                unsafe_allowed_html=True
            )
        with col3:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Detection F1-Score</div>'
                f'<div class="metric-value">{metrics.get("xgb_f1", 0.0):.4f}</div>'
                f'</div>', 
                unsafe_allowed_html=True
            )
        with col4:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Precision / Recall</div>'
                f'<div class="metric-value">{metrics.get("xgb_precision", 0.0):.2f}/{metrics.get("xgb_recall", 0.0):.2f}</div>'
                f'</div>', 
                unsafe_allowed_html=True
            )
            
        st.markdown("#### Hyperparameters Used")
        st.json(metadata.get('config', {}))
    else:
        st.warning("No `metadata.json` found in the root. Train the models using the CLI first to view performance metrics here.")
        st.code("python -m src.cli train --csv-path data/creditcard.csv")

with tab_inference:
    st.markdown("### Run Anomaly & Fraud Predictions")
    st.write("Upload a transaction CSV file (formatted similarly to the Kaggle credit card fraud dataset) to compute reconstruction anomaly scores and final XGBoost fraud probability.")
    
    uploaded_file = st.file_uploader("Choose a CSV file...", type=["csv"])
    
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            st.markdown("#### Input Preview")
            st.dataframe(input_df.head(5))
            
            if st.button("Run AnomalyBoost Pipeline", type="primary"):
                # Save uploaded file temporarily
                temp_input = "temp_uploaded_transactions.csv"
                temp_output = "temp_scored_predictions.csv"
                input_df.to_csv(temp_input, index=False)
                
                with st.spinner("Executing models (preprocessing, reconstruction error estimation, and XGBoost classification)..."):
                    try:
                        from src.pipeline import predict_pipeline
                        # Check if models exist
                        if not os.path.exists("autoencoder_model.h5") or not os.path.exists("xgb_best.joblib"):
                            st.error("Model files (autoencoder_model.h5 / xgb_best.joblib) not found. Run training first.")
                        else:
                            scored_df = predict_pipeline(temp_input, ".", temp_output)
                            
                            st.balloons()
                            st.success("Inference completed!")
                            
                            # Clean up
                            if os.path.exists(temp_input):
                                os.remove(temp_input)
                                
                            st.markdown("### Prediction Results")
                            
                            # Summary metrics
                            total_rows = len(scored_df)
                            frauds = int(np.sum(scored_df['is_fraud_predicted'] == 1))
                            st.write(f"Processed **{total_rows}** transactions. Flagged **{frauds}** as suspected fraudulent cases.")
                            
                            # Results Preview
                            cols_to_show = ['Time', 'Amount', 'recon_error', 'xgb_prob', 'is_fraud_predicted']
                            cols_actual = [c for c in cols_to_show if c in scored_df.columns]
                            if 'Class' in scored_df.columns:
                                cols_actual.append('Class')
                            
                            st.dataframe(scored_df[cols_actual].head(20))
                            
                            # Download Scored Results
                            csv_data = scored_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Complete Scored CSV",
                                data=csv_data,
                                file_name="scored_predictions.encode.csv",
                                mime="text/csv"
                            )
                            
                            # Distribution plots on new data
                            st.markdown("#### Reconstruction Errors & Probabilities Distribution")
                            fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
                            
                            sns.histplot(scored_df['recon_error'], bins=30, kde=True, color='purple', ax=ax[0])
                            ax[0].set_title('Autoencoder Reconstruction Error')
                            ax[0].set_yscale('log')
                            
                            sns.histplot(scored_df['xgb_prob'], bins=30, kde=True, color='teal', ax=ax[1])
                            ax[1].set_title('XGBoost Fraud Probability')
                            ax[1].set_yscale('log')
                            
                            st.pyplot(fig)
                            
                            # Clean up predictions csv
                            if os.path.exists(temp_output):
                                os.remove(temp_output)
                                
                    except Exception as e:
                        st.error(f"Failed to run predictions: {e}")
                        if os.path.exists(temp_input):
                            os.remove(temp_input)
        except Exception as e:
            st.error(f"Error loading CSV: {e}")

with tab_visuals:
    st.markdown("### Interpretability & Explanations")
    st.write("We use SHAP (SHapley Additive exPlanations) to decompose how each feature contributes to the XGBoost classification probability.")
    
    col_img1, col_img2 = st.columns(2)
    
    with col_img1:
        st.markdown("#### SHAP Feature Impact")
        if os.path.exists("shap_summary_plot.png"):
            st.image("shap_summary_plot.png", caption="SHAP Summary Plot (impact of features on fraud prediction).")
        else:
            st.info("SHAP summary plot image (`shap_summary_plot.png`) not found. Run model training to generate it.")
            
    with col_img2:
        st.markdown("#### XGBoost Feature Importance (by Gain)")
        if os.path.exists("xgb_feature_importance.png"):
            st.image("xgb_feature_importance.png", caption="Booster Feature Importance by Gain (reconstruction error is often highly ranked).")
        else:
            st.info("Feature importance plot image (`xgb_feature_importance.png`) not found. Run training to generate it.")

    st.markdown("---")
    st.markdown("#### Model Performance Curves")
    col_curve1, col_curve2 = st.columns(2)
    with col_curve1:
        if os.path.exists("xgb_roc_curve.png"):
            st.image("xgb_roc_curve.png", caption="Supervised Classifier ROC Curve", width=500)
    with col_curve2:
        if os.path.exists("xgb_confusion_matrix.png"):
            st.image("xgb_confusion_matrix.png", caption="Supervised Classifier Confusion Matrix", width=400)
