# Install and load required packages
if (!require(caret)) install.packages("caret")
if (!require(neuralnet)) install.packages("neuralnet")
if (!require(ResourceSelection)) install.packages("ResourceSelection")
if (!require(pROC)) install.packages("pROC")

library(caret)
library(neuralnet)
library(ResourceSelection)
library(pROC)

# STEP 1 - Dataset for the MSD

Input = "
Gender	Marital_status	Education	Handed	Sports_activity	Position_working	Work_environment	MSD	AgeN	ExperienceN	Sleep_hourN	Working_daysN	Working_hoursN	Musculo_problem	BMI
1	1	2	2	1	3	2	2	30	10	6	6	9	2	2.28
1	1	2	2	2	3	2	2	34	6	8	7	7	2	2.90
2	1	2	2	1	3	2	2	32	9	4	2	9	2	2.28
2	1	2	2	1	3	2	2	33	4	5	7	9	2	1.87
2	1	2	2	2	3	2	2	31	2	6	4	4	2	2.01
1	1	1	2	1	3	2	2	31	12	4	3	5	2	2.56
2	2	2	2	2	3	2	2	27	12	7	6	6	1	2.75
2	1	2	2	1	3	1	2	31	6	5	2	6	2	2.28
1	1	1	2	2	3	2	2	28	7	3	4	5	2	1.53
2	1	2	2	2	2	1	1	31	5	6	3	8	2	2.19
1	2	2	2	1	3	2	1	28	12	7	6	5	2	1.90
1	1	2	2	1	3	2	2	32	14	4	7	6	2	1.79
2	1	2	2	2	3	2	1	32	2	6	4	6	2	2.46
1	1	2	2	2	3	2	2	30	14	7	3	7	2	1.80
1	2	2	2	1	3	2	2	28	5	7	5	5	2	1.97
1	1	2	2	2	3	2	2	32	8	4	5	6	2	1.66
2	1	2	1	2	3	2	2	32	4	7	5	6	2	1.63
2	1	2	1	2	3	2	2	33	7	7	4	4	2	2.02
1	1	2	1	2	3	2	2	29	5	5	3	6	2	2.12
2	1	2	1	2	3	2	2	30	6	5	5	10	2	1.13
1	2	1	1	1	3	2	2	29	13	6	6	10	2	2.19
2	1	1	1	1	3	2	2	28	12	8	6	4	2	1.88
1	1	2	1	2	3	2	2	26	5	7	4	9	2	2.82
1	1	2	1	2	3	2	2	35	13	7	3	8	2	1.84
2	1	2	1	1	3	2	2	37	6	4	3	5	2	1.83
2	1	2	1	1	3	2	2	31	14	5	3	10	2	2.04
2	1	1	1	2	3	2	2	32	5	6	5	5	2	2.13
1	1	1	1	2	3	2	1	27	8	7	5	9	2	3.18
2	1	1	2	2	3	2	1	32	3	7	7	6	2	1.59
2	1	2	1	1	3	2	1	30	5	6	3	5	1	2.42
2	1	2	1	2	3	2	2	29	4	6	5	10	2	2.27
1	1	2	2	1	3	2	1	31	3	7	3	6	2	1.56
2	2	1	1	1	3	2	1	25	11	7	6	10	2	1.80
2	1	2	1	2	3	2	2	30	14	6	6	4	2	2.47
1	1	2	1	2	3	2	2	33	2	7	5	7	2	1.73
2	1	2	1	2	3	1	2	30	11	6	6	5	2	1.70
2	1	2	1	1	3	2	2	36	3	5	7	8	2	2.04
2	1	2	1	2	3	1	2	30	6	4	4	5	2	1.90
2	1	2	1	1	3	2	2	34	13	5	6	8	1	1.71
2	2	1	1	1	3	2	2	27	13	7	6	5	1	1.17
1	1	2	2	1	3	2	2	29	13	4	5	8	1	1.29
2	1	1	1	1	3	2	2	32	7	7	4	8	1	3.41
1	1	2	1	1	3	1	2	28	5	7	6	5	1	2.53
2	1	2	1	2	3	2	2	36	9	6	2	6	1	1.30
1	1	1	1	1	3	2	2	28	9	5	7	4	1	1.51
2	1	2	1	1	3	1	2	28	12	7	5	7	2	1.27
2	1	2	1	1	3	2	2	25	8	4	4	7	2	1.67
1	1	2	1	1	3	2	2	37	11	8	7	8	1	2.31
2	2	2	1	1	3	1	1	37	10	7	5	8	1	2.12
2	1	1	1	2	3	1	2	28	14	5	3	7	1	1.64
2	2	1	1	2	3	2	2	28	10	3	3	4	1	2.46
1	1	2	1	1	3	2	2	34	9	4	6	8	1	2.43
2	2	2	2	1	3	2	2	33	6	6	5	5	1	1.56
1	1	2	1	1	3	2	1	32	11	7	3	7	2	1.18
2	1	2	1	1	3	1	1	33	6	8	6	9	1	1.79
1	1	1	1	1	3	2	2	31	10	6	3	5	1	1.41
2	1	2	1	1	3	1	1	34	6	4	5	6	1	2.63
2	1	2	2	2	3	2	2	34	4	8	7	4	1	2.79
1	1	2	1	2	3	1	2	32	4	5	4	7	1	2.36
1	1	1	1	2	3	1	2	36	5	7	3	9	1	2.22
2	1	2	1	2	3	1	2	33	10	5	3	6	1	1.42
1	1	2	1	2	3	1	2	26	7	6	6	8	1	1.35
2	1	2	2	2	3	1	2	37	9	3	3	4	2	1.60
2	1	1	1	1	3	2	2	29	6	6	3	9	1	2.02
1	2	1	1	2	3	2	2	30	4	8	2	9	1	2.41
2	1	2	1	2	3	2	2	36	5	6	5	9	1	1.94
1	1	2	1	2	3	2	2	38	11	6	6	8	2	3.19
2	1	1	1	1	3	2	1	35	4	7	5	9	1	2.36
2	1	2	1	1	3	1	2	33	8	6	3	8	1	1.92
2	1	2	1	2	3	1	2	34	3	6	5	8	1	2.79
2	1	2	1	1	3	2	2	31	7	4	5	9	2	2.46
1	1	2	2	1	3	2	1	31	11	3	6	4	1	1.51
2	1	2	1	1	3	2	2	30	10	5	6	8	1	2.77
2	1	2	1	2	3	2	2	30	3	4	5	6	1	2.28
2	1	1	1	2	3	2	2	31	13	6	6	5	2	2.21
1	1	2	2	1	3	1	2	30	6	4	4	8	1	3.27
2	2	2	1	1	3	1	1	32	14	4	5	9	1	1.99
1	1	2	1	2	3	2	2	31	12	6	2	7	2	2.61
1	1	2	1	2	3	2	2	27	10	3	3	6	1	2.84
2	1	2	1	2	3	2	2	36	6	4	3	8	1	1.20
2	1	2	1	1	3	2	1	35	13	6	5	5	1	2.68
2	1	2	1	2	3	2	1	33	5	6	6	5	1	2.11
2	2	2	1	2	3	1	2	31	10	6	4	10	1	3.64
1	2	2	1	2	3	1	1	26	2	4	6	5	1	2.42
2	2	2	1	2	3	2	2	35	5	4	5	5	2	2.54
1	1	2	2	2	3	2	2	33	6	4	6	6	1	3.37
2	1	2	1	1	3	2	2	34	3	4	5	6	1	2.31
2	1	2	1	1	3	2	2	28	4	5	4	5	1	2.82
1	1	2	1	2	3	2	1	25	13	4	5	5	1	1.75
2	1	2	1	2	3	2	1	33	2	5	5	7	1	1.73
1	1	2	2	2	3	1	1	29	3	4	6	4	1	1.60
1	1	2	1	2	3	1	2	31	4	6	6	8	1	1.82
2	1	2	1	1	3	1	2	31	3	4	5	6	2	1.25
2	1	2	1	2	3	2	2	37	8	8	5	5	1	1.68
2	2	2	1	2	3	2	2	29	4	6	4	5	1	1.83
1	1	2	2	1	3	1	2	35	6	5	6	8	1	2.09
2	1	1	1	2	3	2	2	35	6	7	6	7	1	2.73
2	1	2	1	2	3	2	2	31	11	4	7	8	1	2.86
1	1	2	1	1	3	1	1	30	11	3	5	9	2	1.42
2	1	2	1	1	3	1	1	28	8	7	4	7	1	1.36
2	2	1	2	1	2	2	2	31	4	6	5	5	1	2.52
2	1	2	1	1	2	2	2	35	12	7	6	5	1	1.61
2	1	2	1	1	2	2	2	25	2	4	4	9	2	2.05
2	1	1	1	1	2	2	2	30	5	4	5	8	1	1.26
2	1	2	1	2	2	2	2	28	11	7	6	7	1	3.30
1	1	2	1	2	2	2	2	31	11	5	2	8	2	1.47
2	1	2	1	1	2	1	2	37	14	5	6	6	2	3.13
2	1	2	1	1	2	1	2	27	14	7	5	7	2	1.78
1	1	1	1	1	2	2	2	28	11	6	3	5	2	1.62
1	1	2	1	2	2	2	2	36	13	5	7	6	2	1.37
2	1	2	1	1	2	1	2	31	4	3	7	7	2	2.01
1	1	2	1	2	2	1	2	28	8	3	5	8	2	3.07
2	1	2	1	1	2	1	2	33	4	8	4	8	2	2.58
2	1	1	1	1	2	2	2	25	9	6	5	6	2	1.84
2	1	2	1	1	2	2	2	35	8	6	4	5	2	2.20
1	1	1	1	1	2	2	2	32	4	5	3	6	2	1.98
1	1	2	1	2	2	2	2	33	9	4	3	9	2	1.76
1	2	2	1	1	2	2	2	36	5	4	4	9	2	2.04
1	1	1	1	1	2	2	2	35	6	5	6	5	2	1.24
1	2	2	1	2	2	2	2	28	13	4	3	10	2	1.91
2	1	2	2	1	2	2	2	34	11	7	5	5	2	2.36
2	1	2	1	1	2	2	2	34	6	4	5	5	2	1.59
2	1	2	1	2	2	1	2	31	11	6	4	6	2	2.10
1	1	2	1	2	2	2	2	34	11	6	6	9	2	1.91
1	1	2	1	2	2	2	2	29	4	6	3	10	2	1.13
2	1	2	1	2	2	2	2	30	2	8	3	6	2	2.74
1	1	1	1	2	1	1	2	32	14	6	4	8	2	1.98
2	2	1	1	1	1	2	2	27	14	5	5	6	2	1.57
2	1	2	1	2	1	1	2	31	12	3	6	8	2	1.93
1	1	1	1	2	1	2	1	35	2	7	6	6	2	2.01
2	1	1	1	1	1	1	2	30	8	4	7	8	2	3.11
2	2	2	1	1	1	2	2	32	14	5	3	5	2	2.28
"
data <- read.table(textConnection(Input), header = TRUE)

# ===============================
# STEP 2 - Check Missing Values
# ===============================
print(apply(data, 2, function(x) sum(is.na(x))))

# ===============================
# STEP 3 - Normalize Data
# ===============================
normalize <- function(x) { (x - min(x)) / (max(x) - min(x)) }
maxmindf <- as.data.frame(lapply(data, normalize))

# ===============================
# STEP 4 - Split Dataset
# ===============================
set.seed(123)
index <- sample(1:nrow(maxmindf), 0.7 * nrow(maxmindf))
Training <- maxmindf[index, ]
Testing <- maxmindf[-index, ]
Training$MSD <- as.factor(Training$MSD)

# ===============================
# STEP 5 - Logistic Regression
# ===============================
logistic_model <- glm(MSD ~ Gender + Marital_status + Education + Sports_activity + Work_environment +
                        AgeN + Position_working + ExperienceN + Sleep_hourN +
                        Working_daysN + Working_hoursN + Musculo_problem + BMI,
                      data = Training, family = binomial)

summary(logistic_model)

# ===============================
# STEP 6 - Predictions
# ===============================
logistic_predictions <- predict(logistic_model, newdata = Testing, type = "response")
predicted_classes <- ifelse(logistic_predictions > 0.5, 1, 0)
actual_classes <- Testing$MSD

# ===============================
# STEP 7 - Metrics
# ===============================
confusion_matrix <- table(Predicted = predicted_classes, Actual = actual_classes)
print(confusion_matrix)

accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("✅ Logistic Regression Accuracy:", round(accuracy * 100, 2), "%\n")

MSE_logistic <- mean((as.numeric(as.character(Testing$MSD)) - logistic_predictions)^2)
cat("📉 Logistic Regression MSE:", round(MSE_logistic, 4), "\n")

# Additional Metrics
cm <- confusionMatrix(as.factor(predicted_classes), as.factor(actual_classes))
print(cm)

# ===============================
# STEP 8 - AUC
# ===============================
roc_logistic <- roc(actual_classes, logistic_predictions)
cat("🎯 AUC:", round(auc(roc_logistic), 4), "\n")

# ===============================
# STEP 9 - Hosmer-Lemeshow Test (FIXED)
# ===============================
hl_test <- hoslem.test(as.numeric(as.character(Training$MSD)), fitted(logistic_model), g = 10)
print(hl_test)

# ===============================
# STEP 10 - Brier Score
# ===============================
brier_score_logistic <- mean((logistic_predictions - as.numeric(as.character(Testing$MSD)))^2)
cat("🔵 Brier Score:", round(brier_score_logistic, 4), "\n")

# ===============================
# STEP 11 - LIME (Optional Explainability)
# ===============================
model_type.glm <- function(x, ...) "classification"
predict_model.glm <- function(model, newdata, ...) {
  preds <- predict(model, newdata, type = "response")
  data.frame(`1` = preds, `0` = 1 - preds)
}

Training_lime <- Training %>% mutate_if(is.factor, as.numeric)
Testing_lime  <- Testing %>% mutate_if(is.factor, as.numeric)
x_train_lime <- Training_lime[, !names(Training_lime) %in% "MSD"]
x_test_lime  <- Testing_lime[, !names(Testing_lime) %in% "MSD"]

logistic_explainer <- suppressWarnings(
  lime(x = x_train_lime, model = logistic_model, bin_continuous = TRUE)
)

logistic_lime_exp <- lime::explain(
  x = x_test_lime[1:5, ],
  explainer = logistic_explainer,
  n_labels = 1,
  n_features = 5
)

plot_features(logistic_lime_exp)

# ===============================
# STEP 12 - SHAP (Using iml)
# ===============================
predictor_logistic <- Predictor$new(
  model = logistic_model,
  data = Training[, -which(names(Training) == "MSD")],
  y = as.numeric(as.character(Training$MSD)),
  type = "response"
)

shap_logistic <- Shapley$new(predictor_logistic, x.interest = Testing[1, -which(names(Testing) == "MSD")])
plot(shap_logistic)


#################### Multilayer Perceptron (Neural Network) ####################

# 1. Re-load packages freshly
detach("package:neuralnet", unload = TRUE)
library(neuralnet)

# 2. Ensure target variable is numeric and clean
data$MSD <- as.numeric(as.character(data$MSD))

# 3. Normalize data
normalize <- function(x) (x - min(x)) / (max(x) - min(x))
maxmindf <- as.data.frame(lapply(data, normalize))

# 4. Split dataset
set.seed(123)
index <- sample(1:nrow(maxmindf), 0.7 * nrow(maxmindf))
train_data <- maxmindf[index, ]
test_data <- maxmindf[-index, ]

# 5. Define formula for neuralnet
formula_nn <- as.formula("MSD ~ Gender + Marital_status + Education + Sports_activity +
                          Work_environment + AgeN + Position_working + ExperienceN + Sleep_hourN +
                          Working_daysN + Working_hoursN + Musculo_problem + BMI")

# 6. Train neural network
nn_model <- neuralnet(formula = formula_nn,
                      data = train_data,
                      hidden = c(8, 8, 1),
                      act.fct = "tanh",
                      linear.output = FALSE,
                      stepmax = 5e6)

# 7. Plot model
plot(nn_model)

# 8. Predict
test_inputs <- subset(test_data, select = c("Gender", "Marital_status", "Education", "Sports_activity",
                                            "Work_environment", "AgeN", "Position_working", "ExperienceN",
                                            "Sleep_hourN", "Working_daysN", "Working_hoursN",
                                            "Musculo_problem", "BMI"))

# ✅ MAIN FIX: Avoid conditional test – `class(nn_model)` should be checked with str() if needed
nn_result <- compute(nn_model, test_inputs)

# 9. Convert output
predicted_prob <- nn_result$net.result
predicted_class <- ifelse(predicted_prob > 0.5, 1, 0)
actual_class <- round(test_data$MSD)

# 10. Evaluation
conf_mat <- table(Predicted = predicted_class, Actual = actual_class)
print(conf_mat)

accuracy <- sum(diag(conf_mat)) / sum(conf_mat)
cat("✅ Neural Network Accuracy:", round(accuracy * 100, 2), "%\n")

# Calculate accuracy in percentage
accuracy_in_percent <- (1 - value) * 100
print(paste("Neural Network Accuracy: ", accuracy_in_percent, "%"))

mse <- mean((predicted_prob - test_data$MSD)^2)
cat("📉 Neural Network MSE:", round(mse, 4), "\n")

#Step 16-ROC Curve

# Plot ROC Curve for Logistic Regression
plot(roc_logistic, col = "blue", lwd = 2, main = "ROC Curve for Logistic Regression Model")
text(0.6, 0.4, labels = paste("AUC:", round(auc_logistic, 2)), col = "blue")

# Plot ROC Curve for Hybrid Model
plot(roc_nn, col = "red", lwd = 2, main = "ROC Curve for Hybrid Model")
text(0.6, 0.4, labels = paste("AUC:", round(auc_nn, 2)), col = "red")

# Overlay both ROC Curves
plot(roc_logistic, col = "blue", lwd = 2, main = "ROC Curves for Logistic Regression and Hybrid Model-MSD")
plot(roc_nn, col = "red", lwd = 2, add = TRUE)
legend("bottomright", legend = c("Logistic Regression-MSD", "Hybrid Model-MSD"), col = c("blue", "red"), lwd = 2)
text(0.6, 0.4, labels = paste("AUC MLR:", round(auc_logistic, 2)), col = "blue")
text(0.6, 0.3, labels = paste("AUC Hybrid:", round(auc_nn, 2)), col = "red")

#Step 17-apply McNemar's and Paired T-Test to determine whether the differences in prediction accuracy between the Multiple Logistic Regression (MLR) and the MLR + Multilayer Perceptron (MLP) models are statistically significant#
# Assuming you already have predicted_classes_mlr and predicted_classes_mlp
threshold <- 0.5  # Change this to optimize
predicted_classes_mlr <- ifelse(logistic_predictions > threshold, 1, 0)
predicted_classes_mlp <- ifelse(predicted_probabilities_mlp > threshold, 1, 0)

#Create the McNemar's table
mc_table <- table(MLR = predicted_classes_mlr, MLP = predicted_classes_mlp)
print(mc_table)

#Apply McNemar's test
mcnemar_result <- mcnemar.test(mc_table)

# Print the test result
print(mcnemar_result)

# Ensure the MSD column is a factor for classification
Training$MSD <- as.factor(Training$MSD)

# Set up 3-fold cross-validation
set.seed(123)
train_control <- trainControl(method = "cv", number = 3)

# Perform 3-fold cross-validation for MLR using caret
mlr_cv_model <- train(MSD ~ Gender + Marital_status + Education + Sports_activity + Work_environment + 
                        AgeN + Position_working + ExperienceN + Sleep_hourN + 
                        Working_daysN + Working_hoursN + Musculo_problem + BMI, 
                      data = Training, method = "glm", family = "binomial", 
                      trControl = train_control)

# Perform 3-fold cross-validation for MLP using caret
mlp_cv_model <- train(MSD ~ Gender + Marital_status + Education + Sports_activity + Work_environment + 
                        AgeN + Position_working + ExperienceN + Sleep_hourN + 
                        Working_daysN + Working_hoursN + Musculo_problem + BMI, 
                      data = Training, method = "nnet", linout = FALSE, trace = FALSE, 
                      trControl = train_control)

# Extract accuracy from resampling results
accuracy_mlr_cv <- mlr_cv_model$resample$Accuracy
accuracy_mlp_cv <- mlp_cv_model$resample$Accuracy

# Check the length of the accuracy vectors to ensure they match
print(length(accuracy_mlr_cv))
print(length(accuracy_mlp_cv))

# Perform paired t-test on cross-validated accuracies
paired_ttest_result_cv <- t.test(accuracy_mlr_cv, accuracy_mlp_cv, paired = TRUE)

# Print the paired t-test result
print(paired_ttest_result_cv)

# 🔍 LIME for Neural Network
# ===============================
library(lime)
library(dplyr)

# Define model type and prediction function for neuralnet
model_type.nn <- function(x, ...) "classification"
predict_model.nn <- function(model, newdata, ...) {
  pred <- compute(model, newdata)$net.result
  data.frame(`1` = pred, `0` = 1 - pred)
}

# Prepare numeric data for LIME
Training_lime <- train_data %>% mutate_if(is.factor, as.numeric)
Testing_lime  <- test_data %>% mutate_if(is.factor, as.numeric)
x_train_lime <- Training_lime[, !names(Training_lime) %in% "MSD"]
x_test_lime  <- Testing_lime[, !names(Testing_lime) %in% "MSD"]

# Create explainer (use lime::lime explicitly)
nn_explainer <- lime::lime(
  x = x_train_lime,
  model = nn_model,
  bin_continuous = TRUE
)

# Explain first 5 predictions
nn_lime_exp <- lime::explain(
  x = x_test_lime[1:5, ],
  explainer = nn_explainer,
  n_labels = 1,
  n_features = 5
)

# Plot results
plot_features(nn_lime_exp)

# 🧠 SHAP for Neural Network (using iml)
# ===============================
library(iml)

# Wrap model manually since neuralnet is not native to iml
pred_nn_wrapper <- function(model, newdata) {
  compute(model, newdata)$net.result
}

predictor_nn <- Predictor$new(
  model = nn,
  data = x_train_lime,
  y = Training$MSD,
  predict.function = pred_nn_wrapper,
  type = "response"
)

shap_nn <- Shapley$new(predictor_nn, x.interest = x_test_lime[1, ])
plot(shap_nn)