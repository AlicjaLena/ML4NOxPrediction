"""
Module for downloading datasets from Kaggle using the Kaggle API.
"""

import os
from pathlib import Path
from typing import Optional

import kaggle

def download_kaggle_dataset(
        dataset_name: str,
        download_path: str,
        kaggle_json_path: Optional[str] = None
) -> None:
    """
    Downloads a dataset from Kaggle using the Kaggle API and extracts it to the specified directory.

    :param dataset_name: The name of the Kaggle dataset in the format "username/dataset-name"
    :param download_path: The path to the directory where the dataset
    should be downloaded and extracted
    :param kaggle_json_path: Optional path to the kaggle.json file
    (if not in the default ~/.kaggle directory)
    :raises FileNotFoundError: If the kaggle.json file is not found at the provided path
    :raises ValueError: If the dataset name or download path is invalid
    """
    # Validate inputs
    if not dataset_name:
        raise ValueError("Dataset name cannot be empty.")
    if not download_path:
        raise ValueError("Download path cannot be empty.")

    # Configure the Kaggle API to use a custom path for kaggle.json if provided
    if kaggle_json_path:
        kaggle_json = Path(kaggle_json_path) / "kaggle.json"
        if not kaggle_json.exists():
            raise FileNotFoundError(
                f"kaggle.json file not found in the specified path: {kaggle_json_path}"
                )
        os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_json_path)

    # Ensure the download directory exists
    download_dir = Path(download_path)
    download_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download and extract the dataset
        kaggle.api.dataset_download_files(dataset_name, path=str(download_dir), unzip=True)
        print(f"Dataset '{dataset_name}' downloaded and extracted to: {download_dir}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while downloading the dataset: {e}") from e
