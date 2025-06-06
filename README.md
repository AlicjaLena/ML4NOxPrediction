# 💨 Predicting NOx Emissions in Gas Turbines Using Machine Learning
This project focuses on predicting nitrogen oxides (NOₓ) emissions from gas turbines using supervised machine learning models. The dataset consists of real-world operational measurements collected from a gas turbine over five years, and includes engineered features designed to improve model generalization in dynamic conditions.

## 📊 Dataset

The dataset is publicly available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set). It contains hourly measurements from a gas turbine fueled by natural gas. 

### How to Use the Dataset
Download the .zip archive containing gt_2011.xlsx through gt_2015.xlsx.

Extract all .xlsx files into the ./data directory.

Use the provided preprocessing script (preprocess.py) to load and convert them to .csv.

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
* Convert .xlsx to .csv and split chronologically into train/val/test.
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
* Save trained models (joblib) and predictions on test data.

---

## 🛠️ Tech Stack

* Language: Python 3.8+
* Data Handling: pandas, numpy
* Visualization: matplotlib, seaborn
* ML Models: scikit-learn, XGBoost, TensorFlow (optional for MLP)

---

## 🚀 How to Run
1. Clone the repository:
```
git clone https://github.com/AlicjaLena/ML4NOxPrediction.git
cd ML4NOxPrediction
```
2. Install dependencies:
```
pip install -r requirements.txt
```  
3. Place the csv files (gt_2011.csv to gt_2015.csv) in ./data
4. Preprocess and engineer features:
```
python preprocess.py
```   
5. Train the model (example):
```
python models/train_xgb.py
``` 

### Prerequisites

- Python 3.8+
- Install dependencies using:
  ```bash
  pip install -r requirements.txt
  ```
### 📄 License
Data is available under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/legalcode). This project code is released under the MIT License.
