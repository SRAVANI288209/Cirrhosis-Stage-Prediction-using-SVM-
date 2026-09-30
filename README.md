# Cirrhosis-Stage-Prediction-using-SVM-
Machine learning-based multiclass classification system for predicting liver cirrhosis stages (Stage 1–4) using clinical and laboratory data. The project includes EDA, data preprocessing, SMOTE, multiple ML models, SVM hyperparameter tuning, and model deployment.
# Cirrhosis Stage Prediction Using SVM

## Project Overview

Machine learning-based multiclass classification system for predicting liver cirrhosis disease stages (Stage 1, Stage 2, Stage 3, and Stage 4) using structured clinical and laboratory data.

The project includes Exploratory Data Analysis (EDA), data preprocessing, handling class imbalance using SMOTE, comparison of multiple machine learning algorithms, SVM hyperparameter tuning, cross-validation, and model deployment.

## Problem Statement

Cirrhosis is a chronic liver condition that causes progressive damage and scarring of the liver.

The objective of this project is to develop a machine learning model that classifies patients into Stage 1, Stage 2, Stage 3, or Stage 4 using clinical, demographic, treatment, and laboratory features.

## Dataset

- Total records: 312
- Original columns: 20
- Final input features: 18
- Target variable: Stage
- Classes: Stage 1–4

### Input Features

- N_days
- Age
- Sex
- Drug
- Ascites
- Hepatomegaly
- Spiders
- Edema
- Bilirubin
- Cholesterol
- Albumin
- Copper
- Alk_Phos
- SGOT
- Tryglicerides
- Platelets
- Prothrombin

## Exploratory Data Analysis

The following EDA steps were performed:

- Checked dataset shape, columns, and data types
- Analyzed missing values
- Handled missing values using appropriate imputation
- Checked and handled duplicate values
- Detected and handled outliers using the IQR method
- Performed univariate, bivariate, and multivariate analysis
- Analyzed feature correlations
- Analyzed Stage distribution
- Encoded categorical variables
- Applied feature scaling

## Handling Class Imbalance

SMOTE (Synthetic Minority Over-sampling Technique) was used to handle class imbalance.

SMOTE was applied only to the training data to avoid data leakage.

## Machine Learning Models

The following classification algorithms were evaluated:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- K-Nearest Neighbors (KNN)
- Decision Tree
- Gradient Boosting

## Final Model

### Support Vector Machine (SVM)

SVM was selected as the final model based on the final test performance among the evaluated models.

### Test Performance

- Test Accuracy: 55.56%
- Test Macro F1 Score: 0.5125

## Hyperparameter Tuning

GridSearchCV was used for SVM hyperparameter tuning.

Parameters evaluated:

- C: 0.1, 1, 10, 100
- Gamma: scale, auto, 0.001, 0.01, 0.1
- Kernel: Linear, RBF
- Class Weight: None, Balanced

### Best Parameters

- C = 10
- Kernel = RBF
- Gamma = auto
- Class Weight = None

## Cross-Validation

5-Fold Stratified Cross-Validation was used for model evaluation.

Stratification maintains similar Stage-class proportions across the folds.

### Cross-Validation Results

- Mean CV Accuracy: 68.76%
- CV Standard Deviation: 5.76%

## Project Workflow

Patient Input  
↓  
Data Preprocessing  
↓  
Saved SVM Model  
↓  
Stage Prediction

## Project Files

- `01_data_exploration.ipynb` – Data exploration and analysis
- `cirrhosis.csv` – Dataset
- `app.py` – Application/deployment code
- `index.html` – User interface
- `best_cirrhosis_model.pkl` – Saved machine learning model
- `cirrhosis_flask_pipeline.pkl` – Saved preprocessing/model pipeline
- `requirements.txt` – Required Python libraries
- `README.md` – Project documentation

## Deployment

The trained SVM model was saved using Joblib and can be loaded for prediction without retraining.

The application accepts patient information, processes the input, and predicts the cirrhosis stage.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Joblib
- Flask
- HTML

## Future Enhancements

- Increase the size and diversity of the dataset
- Improve prediction of minority Stage classes
- Explore advanced ensemble and deep learning models
- Perform external validation using an independent dataset

## Disclaimer

This project is developed for educational and machine learning purposes. It is not intended to replace professional medical diagnosis or clinical decision-making.
