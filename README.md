# Life Expectancy Prediction using XGBoost

A machine learning regression project that predicts **life expectancy** for countries using health, economic, and social indicators, powered by an **XGBoost Regressor**.

## 📊 Overview

This project analyzes the WHO Life Expectancy dataset (2938 records, 21 features) covering factors such as mortality rates, immunization coverage, GDP, schooling, and more. An XGBoost regression model is trained to predict life expectancy with high accuracy.

## 🎯 Project Workflow

1. **Import Key Libraries & Datasets** — Load data with `pandas`, `numpy`, `seaborn`, and `matplotlib`.
2. **Data Visualization** — Explore missing values, feature distributions, and correlations via heatmaps and histograms.
3. **Feature Engineering** — One-hot encode categorical features (`Status`) and impute missing values using column means.
4. **Model Training** — Train an `XGBRegressor` on a 70/30 train-test split.
5. **Model Evaluation** — Assess performance using RMSE, MSE, MAE, R², and Adjusted R².

## 🗂️ Dataset

The dataset (`Life_Expectancy_Data.csv`) contains 21 columns, including:

- `Life expectancy` (target variable)
- `Adult Mortality`, `infant deaths`, `under-five deaths`
- `Alcohol`, `GDP`, `Population`, `Schooling`
- Immunization coverage: `Hepatitis B`, `Polio`, `Diphtheria`
- `BMI`, `thinness 1-19 years`, `thinness 5-9 years`
- `Income composition of resources`
- `Status` (Developed / Developing)

## 🛠️ Tech Stack

- Python 3.10
- pandas, numpy
- seaborn, matplotlib (visualization)
- scikit-learn (train/test split, evaluation metrics)
- XGBoost (regression model)

## ⚙️ Installation

```bash
git clone https://github.com/<Manahilch18>/<xgboost-life-expectancy-regressor>.git
cd <xgboost-life-expectancy-regressor>
pip install -r requirements.txt
```

**requirements.txt**
```
pandas
numpy
seaborn
matplotlib
scikit-learn
xgboost
```

## 🚀 Usage

Open and run `main.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab. Make sure `Life_Expectancy_Data.csv` is in the same directory (or update the file path in the notebook).

```python
import xgboost as xgb

model = xgb.XGBRegressor(
    objective='reg:squarederror',
    learning_rate=0.1,
    max_depth=30,
    n_estimators=100
)
model.fit(X_train, y_train)
```

## 📈 Model Performance

| Metric        | Value  |
|---------------|--------|
| Accuracy (R²) | 0.959  |
| RMSE          | 1.875  |
| MSE           | 3.515  |
| MAE           | 1.157  |
| Adjusted R²   | 0.958  |

The model explains ~95.9% of the variance in life expectancy on the held-out test set, indicating strong predictive performance.

## 📁 Project Structure

```
├── main.ipynb              # Main notebook (EDA, feature engineering, model training/eval)
├── Life_Expectancy_Data.csv # Dataset (WHO life expectancy data)
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## 🔮 Future Improvements

- Hyperparameter tuning (GridSearchCV / Optuna) for XGBoost
- Feature importance analysis and SHAP explainability
- Cross-validation instead of a single train/test split
- Handle missing data with more advanced imputation (KNN, iterative imputer)
- Compare against other models (Random Forest, LightGBM, Linear Regression)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙌 Acknowledgements

Dataset sourced from the WHO Global Health Observatory (GHO) life expectancy data.
