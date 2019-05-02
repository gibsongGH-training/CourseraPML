# Practical Machine Learning Course Project
# Predict the manner of exercise

library(caret)

# download files for project
trainfile <- read.csv("C:/gitpost/training.csv", header=TRUE)  #https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
testfile <- read.csv("C:/gitpost/testing.csv", header=TRUE)  #https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

# observe data
str(trainfile)  # 19,622 obs., 160 variables, many NAs and repeating figures
summary(trainfile)  # many columns have 19,216 blanks

# Remove poor predictors with very little variability, per Covariate Creation lecture
nsvTrain <- nearZeroVar(trainfile) 
trainfileSub1 <- trainfile[,-nsvTrain]  # 100 variables

# address the NAs
missingTrain <- apply(is.na(trainfileSub1), 2, sum) > 19000 # TRUE if column has more than 19,000 NAs

trainfileSub2 <- trainfileSub1[,which(missingTrain==FALSE)] # 59 variables

# The first six columns remaining are related to row counts, names, timestamps and windows, not motions - exclude
trainfileSub3 <- trainfileSub2[,-c(1:6)]  # 53 variables

# Reproduce training data preparation steps with testing data 
nsvTest <- nearZeroVar(testfile) 
testfileSub1 <- testfile[,-nsvTest]
missingTest <- apply(is.na(testfileSub1), 2, sum) > 19000
testfileSub2 <- testfileSub1[,which(missingTest==FALSE)]
testfileSub3 <- testfileSub2[,-c(1:6)]

set.seed(1234)  # for reproduceable results

# Set aside validation data from training data
inTrain <- createDataPartition(y=trainfileSub3$classe, p=0.7, list = FALSE)

training <- trainfileSub3[inTrain,]
validation <- trainfileSub3[-inTrain,]

# Decision Tree, from Predicting with Trees lecture, poor accuracy 48.9%
modelFit <- train(classe ~., data=training, method = "rpart")  
# Follow Caret Package lecture
predictions <- predict(modelFit, newdata=validation)  
confusionMatrix(validation$classe, predictions)  

# Random Forest, 3 hours 40 min run, 99.4% accuracy
modelFit <- train(classe ~., data=training, method="rf")
predictions <- predict(modelFit, newdata=validation)
confusionMatrix(validation$classe, predictions)

# gbm boosting method, 96.5% accuracy
modelFit3 <- train(classe ~., data=training, method="gbm")
predictions3 <- predict(modelFit3, newdata=validation)
confusionMatrix(validation$classe, predictions3)

# Run RF model on test data
final <- predict(modelFit, newdata=testfileSub3)
final
