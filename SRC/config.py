"""
Configuration module for dataset and file paths.

This module loads environment variables from a `.env` file
and sets up default paths for datasets, models, results, and logs.

Environment Variables:
- DATA_DIR: Directory where datasets are stored (default: `./data`).
- KAGGLE_JSON_PATH: Path to the Kaggle API authentication file (default: `~/.kaggle`).
- DATASET_NAME: The name of the dataset on Kaggle (default: `sjagkoo7/fuel-gas-emission`).
- MODEL_DIR: Directory for saving trained models (default: `./models`).
- RESULTS_DIR: Directory for storing analysis results (default: `./results`).
- LOGS_DIR: Directory for storing log files (default: `./logs`).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Dataset configuration
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
KAGGLE_JSON_PATH = Path(os.getenv("KAGGLE_JSON_PATH", "~/.kaggle")).expanduser()
DATASET_NAME = os.getenv("DATASET_NAME", "sjagkoo7/fuel-gas-emission")

# Additional directories
MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "./results"))
LOGS_DIR = Path(os.getenv("LOGS_DIR", "./logs"))
