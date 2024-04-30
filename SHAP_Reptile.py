import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import shap

# Load the dataset
data_path = 'E:\\pythonProject\\paper\\model_data.xlsx'
# data_path = 'E:\\pythonProject\\paper\\scenario_data.xlsx'
data = pd.read_excel(data_path)

# Split the dataset into input variables X and output variable y
X = data.drop(data.columns[-1], axis=1)
y = data[data.columns[-1]]

# Train a Random Forest Regressor model, setting a random seed for consistent results
reptile_model = RandomForestRegressor(n_estimators=100, random_state=42)
reptile_model.fit(X, y)

# Calculate SHAP values
explainer = shap.TreeExplainer(reptile_model)
shap_values = explainer.shap_values(X)

# Plot a SHAP bar plot
shap.summary_plot(shap_values, X, plot_type="bar", max_display=10)
