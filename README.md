
# MSD Prediction using Logistic Regression and Neural Network (MLP)

This project implements two models — Multiple Logistic Regression (MLR) and a Multilayer Perceptron (MLP) — for predicting Musculoskeletal Disorder (MSD) based on various workplace and individual factors. The models are evaluated using accuracy, MSE, ROC-AUC, Hosmer-Lemeshow test, Brier Score, and are further explained using LIME and SHAP.

## Features
- Logistic Regression with detailed metrics
- Multilayer Perceptron using `neuralnet`
- Explainability using LIME and SHAP
- Evaluation via cross-validation, ROC curve, McNemar’s Test, and paired t-test

## Requirements
This project uses **R**. You must have the following packages:
- `caret`
- `neuralnet`
- `pROC`
- `ResourceSelection`
- `lime`
- `iml`
- `dplyr`

Install them using:

```r
install.packages(c("caret", "neuralnet", "pROC", "ResourceSelection", "lime", "iml", "dplyr"))
```

## Usage
Run the R script `msd_prediction.R` in RStudio or any R environment.

## Contents
- Logistic Regression and MLP modeling
- Dataset normalization and splitting
- Evaluation metrics and visualizations
- Explainable AI using LIME and SHAP
- Statistical tests for model comparison

## Author
Generated automatically from provided source code.
