"""
Main pipeline script to run full NOx emission analysis using ML models.
Includes data preprocessing, model training, comparison and prediction.
"""

import logging
import sys

# Adjust imports if modules are reorganized into packages
from features import preprocess
from models.train_rf import main as train_rf_main
from models.train_xgb import main as train_xgb_main
from models.train_mlp import main as train_mlp_main
from models.compare_models import main as compare_main
from models.predict import main as predict_main

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def run_pipeline():
    try:
        logging.info("Step 1: Data preprocessing")
        preprocess.main()

        logging.info("Step 2: Train Random Forest model")
        train_rf_main()

        logging.info("Step 3: Train XGBoost model")
        train_xgb_main()

        logging.info("Step 4: Train MLP model")
        train_mlp_main()

        logging.info("Step 5: Compare model performance")
        compare_main()

        logging.info("Step 6: Predict NOx emissions on test set")
        predict_main()

        logging.info("✅ Full analysis completed successfully. Results saved in /results and /artifacts.")

    except Exception as e:
        logging.error("❌ Pipeline terminated due to an error.")
        logging.exception(e)


if __name__ == "__main__":
    run_pipeline()
