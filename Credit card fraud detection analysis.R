#CodeClause Internship Task-2

#Aim 
#Detect fraudulent credit card transactions using advanced machine learning
#techniques.

#Description
#Apply advanced classification algorithms for identifying potential fraudulent
#activities.

#Install packages
install.packages("DMwR")
install.packages("xgboost")
install.packages("keras")
install.packages("pROC")

# Load necessary libraries
library(randomForest)
library(tidyverse)
library(caret)
library(DMwR)  # For SMOTE
library(xgboost)
library(keras)
library(pROC)

#Load the data
credit_card<-read.csv("C:\\Users\\User\\Documents\\Data Science\\Internship\\CodeClause Internship\\Task-2\\creditcard.csv")

#View the data
View(credit_card)

#Column names of data
colnames(credit_card)

#Observing summary
summary(credit_card)

#Observing no. of rows & columns of the dataset
dim(credit_card)

#Checking for missing value
any(is.na(credit_card))

#Remove NA values
credit_card<-na.omit(credit_card)

#Again check for missing value
any(is.na(credit_card))


#Converting Class column to factor
credit_card$Class<-as.factor(credit_card$Class)


#Scale the numerical columns (excluding 'Class' and 'Time' if needed)
credit_card_scaled <- credit_card %>%
  mutate(across(starts_with("V"), scale), Amount = scale(Amount))

# Split data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(credit_card_scaled$Class, p = 0.8, list = FALSE)
train_data <- credit_card_scaled[trainIndex, ]
test_data <- credit_card_scaled[-trainIndex, ]


# Train a Random Forest model
random_forest_model <- randomForest(Class ~ ., data = head(train_data,2500), importance = TRUE, ntree = 10)

# Fit a logistic regression model (GLM) on the training data
glm_model <- glm(Class ~ ., data = train_data, family = binomial)


# Predict on the test set for random forest model
Random_forest_predicted <- predict(random_forest_model, test_data)
Random_forest_predicted

# Predict on the test set using the logistic regression model
glm_pred_probs <- predict(glm_model, newdata = test_data, type = "response")
glm_pred_class <- ifelse(glm_pred_probs > 0.5, 1, 0)  # Convert probabilities to binary classes
glm_pred_class

#Convert as factor
glm_pred_class <- as.factor(glm_pred_class)

#Original values
observed<-credit_card_scaled$Class
observed

#Check accuracy for GLM
accracy_glm<-mean(glm_pred_class==observed)
accracy_glm

#Check accuracy for Random forest
accuracy_random_forest<-mean(observed==Random_forest_predicted)
accuracy_random_forest



# Calculate AUC-ROC for GLM
roc_glm <- roc(as.numeric(test_data$Class) - 1, glm_pred_probs)  # Convert factors to numeric
auc_score_glm <- auc(roc_glm)

# Calculate AUC-ROC for Random forest
rf_pred_probs <- predict(random_forest_model, test_data, type = "prob")[, 2]  # Probabilities for Class "1"
roc_random_forest <- roc(as.numeric(test_data$Class) - 1, rf_pred_probs)  # Convert factors to numeric
auc_score_random_forest <- auc(roc_random_forest)

#Print AUC-ROC score for GLM & Random forest
cat(paste("AUC-ROC for GLM is :", auc_score_glm, "\nAUC-ROC for Random Forest is :", auc_score_random_forest))



# Convert to matrix format as required by XGBoost
X <- as.matrix(credit_card_scaled %>% select(-Class))
y <- as.numeric(credit_card_scaled$Class) - 1  # Convert factors to 0/1

# Split into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
train_X <- X[trainIndex, ]
train_y <- y[trainIndex]
test_X <- X[-trainIndex, ]
test_y <- y[-trainIndex]

# Step 3: Model Training - XGBoost
# Set XGBoost parameters for classification
params <- list(
  objective = "binary:logistic",  # Binary classification
  eval_metric = "auc",            # Use AUC as the evaluation metric
  max_depth = 6,                  # Depth of each tree
  eta = 0.3,                      # Learning rate
  gamma = 1,                      # Minimum loss reduction
  subsample = 0.8,                # Subsampling of rows
  colsample_bytree = 0.8          # Subsampling of columns
)

# Train XGBoost model
dtrain <- xgb.DMatrix(data = train_X, label = train_y)
dtest <- xgb.DMatrix(data = test_X, label = test_y)
xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 100, watchlist = list(train = dtrain), verbose = 0)

# Step 4: Predict and Evaluate the Model
# Predict on test set
pred <- predict(xgb_model, dtest)
pred_class <- as.factor(ifelse(pred > 0.5, 1, 0))
pred_class

#Accuracy
Accuracy<-mean(pred_class==credit_card_scaled$Class)
Accuracy

# Calculate AUC-ROC
roc_obj <- roc(test_y, pred)
auc_score <- auc(roc_obj)
print(paste("AUC-ROC:", auc_score))

















































