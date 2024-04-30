import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the data
data_path = 'E:\\pythonProject\\paper\\model_data.xlsx'
# data_path = 'E:\\pythonProject\\paper\\scenario_data.xlsx'
df = pd.read_excel(data_path)

# Data preprocessing
X = df.iloc[:, :-1]   # Select all columns except the last one as features
y = df.iloc[:, -1]    # Select the last column as the target label

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=44)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model and the parameter grid
rf = RandomForestClassifier(random_state=44)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Set up grid search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)

# Fit the grid search
grid_search.fit(X_train_scaled, y_train)

# Best parameter combination
print("Best Parameters:", grid_search.best_params_)

# Predict using the model with the best parameters
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with best parameters:", accuracy)
