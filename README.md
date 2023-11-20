# Medical-Insurance-Price-Prediction-using-Machine-Learning
## A project to provide insights into the key factors that contribute to higher insurance costs and help the company make more informed decisions regarding pricing and risk assessment.
# Objective:
### The goal is to predict medical insurance expenses based on various factors such as age, sex, BMI, smoking status, number of children, and region.

# Steps Taken:

## 1.]Data Import and Exploration:

### Loaded the medical insurance dataset containing information about 1338 data points with 6 independent features and 1 target feature (charges).Conducted exploratory data analysis (EDA) to understand relationships between different features and the target variable.

## 2.]Data Preprocessing:

### a. Checked for and handled duplicates in the dataset.
### b. Identified and treated outliers in the BMI column using the IQR method.
### c. Explored the distribution of continuous data in the age and BMI columns.
### d. Encoded discrete categorical data (sex, BMI, region) for model prediction.

## 3.]Model Development:

### a. Explored multiple machine learning models (Linear Regression, SVR, Random Forest Regressor, Gradient Boosting Regressor, XGBoost Regressor).
### b. Utilized cross-validation and grid search for hyperparameter tuning.
### c. Identified XGBoost as the best-performing model based on evaluation metrics.

## 4.]Feature Importance:

### Examined feature importances of the final XGBoost model to understand which features contribute most to predictions.

## 5.]Final Model:

### a. Trained the final XGBoost model with optimized hyperparameters.
### b. Dropped less important features to simplify the model.

## 6.]Prediction on New Data:

### Demonstrated how to use the trained model to predict medical insurance charges for new data.

## Conclusion:
### The XGBoost model, after thorough exploration and tuning, was identified as the most effective for predicting medical insurance charges based on the given dataset. The model's feature importances provide insights into the factors that significantly influence insurance costs.
