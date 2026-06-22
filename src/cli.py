import argparse
import sys
import logging
from src.pipeline import train_pipeline, predict_pipeline

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger("src.cli")
    
    parser = argparse.ArgumentParser(
        description="AnomalyBoost: Hybrid Autoencoder + XGBoost Credit Card Fraud Detection Pipeline"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the hybrid fraud detection pipeline")
    train_parser.add_argument("--csv-path", required=True, help="Path to credit card transaction CSV dataset")
    train_parser.add_argument("--output-dir", default=".", help="Directory to save trained model files and outputs")
    train_parser.add_argument("--vae", action="store_true", help="Use Variational Autoencoder (VAE) instead of Dense Autoencoder")
    train_parser.add_argument("--pca", action="store_true", help="Apply PCA dimensionality reduction")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs for the Autoencoder")
    train_parser.add_argument("--batch-size", type=int, default=256, help="Batch size for Autoencoder training")
    train_parser.add_argument("--no-cv", action="store_true", help="Disable cross-validation for XGBoost training")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict fraud probabilities for transaction data")
    predict_parser.add_argument("--csv-path", required=True, help="Path to input transaction CSV dataset")
    predict_parser.add_argument("--model-dir", default=".", help="Directory containing trained model artifacts")
    predict_parser.add_argument("--output-path", required=True, help="Path to save scored prediction output CSV")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    try:
        if args.command == "train":
            config_override = {
                'use_vae': args.vae,
                'use_pca': args.pca,
                'autoencoder_epochs': args.epochs,
                'autoencoder_batch_size': args.batch_size,
                'xgb_use_cv': not args.no_cv
            }
            logger.info("Executing training pipeline...")
            meta = train_pipeline(args.csv_path, args.output_dir, config_override)
            logger.info(f"Training completed. Results metadata saved. Final metrics: {meta.get('metrics', {})}")
            
        elif args.command == "predict":
            logger.info("Executing inference pipeline...")
            predict_pipeline(args.csv_path, args.model_dir, args.output_path)
            logger.info(f"Predictions completed. Scored data written to: {args.output_path}")
            
    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
