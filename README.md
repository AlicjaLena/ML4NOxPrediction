# 💨 Predicting NOx Emissions in Gas Turbines Using Machine Learning
This project focuses on predicting nitrogen oxides (NOₓ) emissions from gas turbines using supervised machine learning models. The dataset consists of real-world operational measurements collected from a gas turbine over five years, and includes engineered features designed to improve model generalization in dynamic conditions.

## 📊 Dataset

The dataset is publicly available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set). It contains hourly measurements from a gas turbine fueled by natural gas. 

### How to Use the Dataset
Download the .zip archive containing gt_2011.xlsx through gt_2015.xlsx.

Extract all .xlsx files into the ./data directory.

Use the provided preprocessing script (features/preprocess.py) to load and convert them to .csv.

### Original Features

| Column | Description                                    |
| ------ | ---------------------------------------------- |
| AT     | Ambient Temperature (°C)                       |
| AP     | Ambient Pressure (mbar)                        |
| AH     | Ambient Humidity (%)                           |
| AFDP   | Air Filter Differential Pressure (mbar)        |
| GTEP   | Gas Turbine Exhaust Pressure (mbar)            |
| TIT    | Turbine Inlet Temperature (°C)                 |
| TAT    | Turbine After Temperature (°C)                 |
| CDP    | Compressor Discharge Pressure (mbar)           |
| TEY    | Turbine Energy Yield (MWh)                     |
| CO     | Carbon Monoxide Emissions (mg/m³)              |
| NOx    | Nitrogen Oxide Emissions (mg/m³) ⬅️ **Target** |

---

## 🔍 Project Workflow

1. Data Preprocessing
* Import .csv files and split chronologically into train/val/test.
* Standardize and clean sensor data.

2. Feature Engineering
* Compute synthetic features such as:
* Energy_Efficiency = TEY / GTEP
* Combustion_Efficiency = TIT - TAT
* Humidity_Temperature_Ratio = AH / AT

3. Model Training
* Train and compare multiple models: Random Forest, XGBoost, MLP.
* Use grid search and cross-validation for hyperparameter tuning.

4. Evaluation
* Metrics: MAE, RMSE, R².
* Visualizations: feature correlation, distributions, prediction plots.

5. Export
* Save trained models (in joblib format) and predictions on the test data.

---

## 🛠️ Tech Stack

* Language: Python 3.8+
* Data Handling: pandas, numpy
* Visualization: matplotlib, seaborn
* ML Models: scikit-learn, XGBoost, TensorFlow (optional for MLP)

---

## 🚀 How to Run
🔁 Full Analysis Reproduction Guide
1. Clone the repository:
```
git clone https://github.com/AlicjaLena/ML4NOxPrediction.git
cd ML4NOxPrediction
```
2. Install dependencies:
Before installing dependencies, it's recommended to isolate the environment:
<details> <summary><strong>Windows (CMD / PowerShell)</strong></summary>
```
python -m venv venv
venv\Scripts\activate
```
</details> <details> <summary><strong>macOS / Linux (bash / zsh)</strong></summary>
 
```
python3 -m venv venv
source venv/bin/activate
```
</details>

After activation, install the dependencies:

```
pip install -r requirements.txt
```  
3. Place the csv files (gt_2011.csv to gt_2015.csv) in ./data
4. Run preprocessing to convert .xlsx to .csv and split into train/val/test:
```
python features/preprocess.py
```   
5. Train models:
  * XGBoost
  ```
  python models/train_xgb.py
  ```
  * Random Fores:
  ```
  python models/train_rf.py
  ```
  * MLP
  ```
  python models/train_mlp.py
  ``` 
6. Compare models
```
python models/compare_models.py
```
7. Predict on Test Set
```
python models/predict.py
```
8. View Results
* Prediction plots: in ./results/figures/
* Predictions: ./results/NOX_predictions_from_test.csv
* Trained models: ./artifacts/

💡 Alternative startup method
1. Clone the repository, set up the execution environment, and install dependencies as in the previous step.
2. Place the csv files (gt_2011.csv to gt_2015.csv) in ./data
3. Run main.py:
```
python main.py
```

### Prerequisites

- Python 3.8+
- Install dependencies using:
  ```bash
  pip install -r requirements.txt
  ```
### 📄 License
Data is available under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/legalcode). This project code is released under the MIT License.
