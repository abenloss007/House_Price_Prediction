# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

#Jets adding
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV

# %%
#import combined data from csv
#data = pd.read_csv("E:\\6242Project\\FeaturesDataset\\philly_property_data.csv")
#data = pd.read_csv("E:\\6242Project\\FeaturesDataset\\washington_property_data.csv")
#data = pd.read_csv("E:\\6242Project\\FeaturesDataset\\philly_enriched.csv")
data = pd.read_csv("C:\\Users\\easto\\OneDrive\\Desktop\\College\\Masters\\Semester 2\\Data and Visual Analytics\\Local Project Data\\philly_final_dataset.csv")



# %%
y = data['sale_price']
data_Features = data.drop(columns=['market_value', 'sale_price'])
#data_hot = data_Features.drop(columns=['sale_date'])
#data_hot = data_Features.drop(columns=['sale_date','view_scale','heater_scale','central_air','garage_spaces','has_central_air', 'has_type_heater'])
data_hot = data_Features
data_hot = pd.get_dummies(data_hot, drop_first=True)


# %%
#train test split
#X_train, X_test, y_train, y_test = train_test_split(data10_hot, y, test_size=0.3, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(data_hot, y, test_size=0.3, random_state=1)

# Standardize full feature set for Elastic Net variable selection
scaler_full = StandardScaler()
X_train_scaled_full = scaler_full.fit_transform(X_train)
X_test_scaled_full = scaler_full.transform(X_test)

# %%
# Variable selection with Elastic Net
elasticNet = ElasticNetCV(l1_ratio=0.5, cv=5)
elasticNet.fit(X_train_scaled_full, y_train)

# Get the selected features
coefficients = elasticNet.coef_
selected_features = data_hot.columns[coefficients != 0]
print("Selected features:")
print(selected_features)

# %%
print("--------------------------------------End of Variable Selection--------------------------------------")

# %%
#Instiantiate model
LMmodel = LinearRegression()

#fit model to training data. Using only selected features
X_train_selected = X_train[selected_features]

LMmodel.fit(X_train_selected, y_train)

#predict on test data
x_test_selected = X_test[selected_features]
y_pred = LMmodel.predict(x_test_selected)

print("Predictions:")
print(y_pred)

#calculate RMSE and R2
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:")
print(RMSE)
print("R2:")
R2 = r2_score(y_test, y_pred)
print(R2)

#Access coefficients
coefficients = LMmodel.coef_
print("Coefficients:")
print(coefficients)

#Access intercept
intercept = LMmodel.intercept_
print("Intercept:")
print(intercept)

# %%
#Instantiating Random Forest Model
RFmodel = RandomForestRegressor(n_estimators=50, random_state=1,n_jobs=-1 ,verbose=1)

#RF model fit
RFmodel.fit(X_train, y_train)

#Sanity check
print("Progress...")

#Predicting on test data
y_pred_RF = RFmodel.predict(X_test)
print("Predictions:")
print(y_pred_RF)

# Calculate RMSE and R2
RMSE = np.sqrt(mean_squared_error(y_test, y_pred_RF))
print("RMSE:")
print(RMSE)
print("R2:")
R2 = r2_score(y_test, y_pred_RF)
print(R2)

# Access feature importances (since Random Forest models provide feature importance)
feature_importances = RFmodel.feature_importances_
print("Feature Importances:")
print(feature_importances)

# %%
# Instantiate Gradient Boosting model

'''
GBmodel = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.15,
    max_depth=4,
    random_state=1,
    verbose=1
)
'''
GBmodel = GradientBoostingRegressor(
    subsample= 1.0, n_estimators= 500,
    min_samples_split= 2, min_samples_leaf= 1,
    max_features= None, max_depth= 4, learning_rate= 0.15
)

# Fit the model on untransformed target
GBmodel.fit(X_train, y_train)

# Predict on test data
y_pred_GB = GBmodel.predict(X_test)

# Evaluate performance
RMSE_GB = np.sqrt(mean_squared_error(y_test, y_pred_GB))
R2_GB = r2_score(y_test, y_pred_GB)

print("Predictions:")
print(y_pred_GB)
print("RMSE (Gradient Boosting):", RMSE_GB)
print("R² (Gradient Boosting):", R2_GB)

# Access feature importances
feature_importances_GB = GBmodel.feature_importances_
print("Feature Importances (Gradient Boosting):")
print(feature_importances_GB)

# %%
# Define parameter distribution
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5],
    'subsample': [0.8, 1.0],
    'max_features': ['sqrt', 'log2', None]
}

# Instantiate base model
gbr = GradientBoostingRegressor(random_state=1)

# Set up randomized search
rand_search = RandomizedSearchCV(
    estimator=gbr,
    param_distributions=param_dist,
    n_iter=50,               # Number of random combinations to try
    scoring='r2',            # Optimize for R² score
    cv=3,                    # 3-fold cross-validation
    verbose=2,
    n_jobs=-1,               # Use all available cores
    random_state=42
)

# Fit to training data
rand_search.fit(X_train, y_train)

# Print best parameters and score
print("Best Parameters Found:")
print(rand_search.best_params_)
print("Best Cross-Validated R² Score:")
print(rand_search.best_score_)

# Evaluate on test set
best_gbr = rand_search.best_estimator_
y_pred_best = best_gbr.predict(X_test)

rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
r2_best = r2_score(y_test, y_pred_best)

print("\nTest Set Evaluation")
print("RMSE:", rmse_best)
print("R²:", r2_best)

#with open("modelSale.pkl", "wb") as f:
    #import pickle
joblib.dump(best_gbr, 'modelSale.pkl')



