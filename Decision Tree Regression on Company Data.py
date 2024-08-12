# Decision Tree Regression on Company Data

"""
Problem Statement:
A cloth manufacturing company is interested in understanding which segments or attributes lead to high sales.
Approach:
- Build a decision tree with 'Sales' as the target variable and all other variables as independent variables in the analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import tree

# Display all columns
pd.set_option('display.max_columns', None)

# Load the data
df = pd.read_csv(r"C:\Users\das.su\OneDrive - GEA\Documents\PDF\Machine Learning\BIT ML, AI and GenAI Course\Company_Data.csv")

# Explore the data
df.info()
df.isnull().sum()

# Visualize relationships between features
sns.pairplot(data=df, hue='ShelveLoc')
plt.show()

# Encode categorical variables
label_encoders = {}
for column in ['ShelveLoc', 'Urban', 'US']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Define features and target variable
X = df.drop(columns=['Sales'])
y = df['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)

# Visualize the decision tree
plt.figure(figsize=(15, 12))
tree.plot_tree(dt_regressor, filled=True)
plt.show()

# Make predictions on the test set
y_pred = dt_regressor.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.4f}')
print(f'R-squared: {r2:.4f}')

# Hyperparameter Tuning using GridSearchCV
print('\nHyperparameter Tuning')
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}

grid_search = GridSearchCV(estimator=DecisionTreeRegressor(random_state=42), 
                           param_grid=param_grid, 
                           cv=5, 
                           n_jobs=-1, 
                           verbose=1, 
                           scoring='r2')

# Fit the model with GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters from Grid Search
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# Train the best model
best_dt_model = grid_search.best_estimator_

# Predictions with the best model
y_pred_best = best_dt_model.predict(X_test)

# Evaluate the tuned model
mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)
print(f'Best Mean Squared Error: {mse_best:.4f}')
print(f'Best R-squared: {r2_best:.4f}')

# Re-draw the decision tree based on the hyperparameter tuning
print('Re-draw the decision tree based on the hyperparameter tuning')
plt.figure(figsize=(15, 12))
tree.plot_tree(best_dt_model, filled=True, feature_names=X.columns, rounded=True)
plt.title('Decision Tree after Hyperparameter Tuning')
plt.show()

# Feature importances
feature_importances = best_dt_model.feature_importances_

# Sorting features by importance
features_sorted = sorted(zip(X.columns, feature_importances), key=lambda x: x[1], reverse=True)
print("\nFeatures sorted by importance:")
for name, importance in features_sorted:
    print(f'{name}: {importance:.4f}')

# Test the model with a single entry (first row)
single_entry_df = pd.DataFrame([X.iloc[0, :]], columns=X.columns)
predicted_sales = best_dt_model.predict(single_entry_df)
print(f'\nPredicted Sales for the first entry: {predicted_sales[0]:.4f}')


