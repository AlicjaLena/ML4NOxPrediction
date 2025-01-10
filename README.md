# Predicting-CO-and-NOx-Emissions-Using-Deep-Learning
This project uses sensor data from a gas turbine to predict emissions of Carbon Monoxide (CO) and Nitrogen Oxides (NOx) using a deep learning approach. The dataset contains 11 features related to the operating conditions of the turbine, which serve as inputs to the model.

## 📊 Dataset

The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/sjagkoo7/fuel-gas-emission/data) and contains hourly aggregated sensor measurements from a gas turbine. It includes the following features: 

- **AT**: Ambient Temperature
- **AP**: Ambient Pressure
- **AH**: Ambient Humidity
- **AFDP**: Air Filter Differential Pressure
- **GTEP**: Gas Turbine Exhaust Pressure
- **TIT**: Turbine Inlet Temperature
- **TAT**: Turbine After Temperature
- **CDP**: Compressor Discharge Pressure
- **TEY**: Turbine Energy Yield
- **CO**: Carbon Monoxide (Target 1)
- **NOx**: Nitrogen Oxides (Target 2)


---

## 🔍 Project Workflow

1. **Data Exploration and Preprocessing**
   - Analyze data distribution, check for missing values, and visualize key features.
   - Standardize and normalize data for deep learning.

2. **Model Design and Training**
   - Build a neural network using TensorFlow/Keras.
   - Train the model to predict two targets: CO and NOx emissions.

3. **Evaluation**
   - Evaluate the model performance using metrics like Mean Squared Error (MSE) and R².
   - Visualize predictions vs. actual values.

4. **Optimization and Deployment**
   - Tune hyperparameters using techniques like Grid Search or Bayesian Optimization.
   - Optionally deploy the model using Flask/Streamlit for interactive prediction.

---

## 🛠️ Tech Stack

- **Programming Language**: Python
- **Libraries**:
  - Data Processing: Pandas, NumPy
  - Visualization: Matplotlib, Seaborn
  - Deep Learning: TensorFlow/Keras
- **Optional Deployment**: Flask, Streamlit

---

## 🚀 How to Run

### Prerequisites

- Python 3.8+
- Install dependencies using:
  ```bash
  pip install -r requirements.txt
