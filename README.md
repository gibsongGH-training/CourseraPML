# CourseraPML
Practical Machine Learning

Greg Gibson
May 2nd 2019

Objective
Predict the manner in which subjects performed their exercise based on data captured by motion accelerometers.  The Classe variable outcome can be A, B, C, D, E based on performance.

Data Preparation
Download and connect to provided data with read.csv function.  Load library caret.  Observe data with str and summary functions.  There are 160 variables and 19,622 observations.  Many variables appear to have numerous blanks, 19,216, represented as NA.
  str(trainfile)  # 19,622 obs., 160 variables, many NAs and repeating figures
  summary(trainfile)  # many columns have 19,216 blanks

Reduce the number of ineffective variables.  First, per the Covariate Creation lecture, utilize the nearZeroVar function to remove variables with little variability and least helpful toward prediction.  This removed 60 variables and left 100.  
  nsvTrain <- nearZeroVar(trainfile) 
  trainfileSub1 <- trainfile[,-nsvTrain]  # 100 variables

Next, address the NAs.  The is.na function will identify NAs as True/False, and I use an apply function to sum each column variable.  If the sum of NAs is greater than 19,000, that variable is True.  I then create a subfile with only the variables that are False, meaning they do not have over 19,000 NAs.  This removed 41 additional variables and left 59.       
  missingTrain <- apply(is.na(trainfileSub1), 2, sum) > 19000 
  # TRUE if column has more than 19,000 NAs
  trainfileSub2 <- trainfileSub1[,which(missingTrain==FALSE)] # 59 variables

A minor adjustment.  The first six remaining variables appear to be observation number, names, timestamps and windows, not related to our exercise prediction.  The last sub file removes these and has 53 remaining variables.
  trainfileSub3 <- trainfileSub2[,-c(1:6)]  # 53 variables

The same steps are conducted to prepare the provided testing data.  The training data is further divided into training and validation sets for cross-validation.  And set seed for reproducibility.
  set.seed(1234)  
  inTrain <- createDataPartition(y=trainfileSub3$classe, p=0.7, list = FALSE)
  training <- trainfileSub3[inTrain,]
  validation <- trainfileSub3[-inTrain,]

Model Training
As noted in the lecture series, Random Forest and Boosting are considered very effective and popular prediction models.  I’ll use those, but first apply a single Decision Tree for comparison, per the Predicting with Trees and Caret Package lectures.
	Decision Tree
  modelFit <- train(classe ~., data=training, method = "rpart")  
  predictions <- predict(modelFit, newdata=validation)  
  confusionMatrix(validation$classe, predictions)

            Reference
  Prediction    A    B    C    D    E
           A 1530   35  105    0    4
           B  486  379  274    0    0 
           C  493   31  502    0    0
           D  452  164  348    0    0
           E  168  145  302    0  467

  Overall Statistics

                 Accuracy : 0.489           
                   95% CI : (0.4762, 0.5019)
      No Information Rate : 0.5317          
      P-Value [Acc > NIR] : 1               

                    Kappa : 0.3311          
  Mcnemar's Test P-Value : NA              	
The Decision Tree was fast, but the prediction accuracy was worse than a coin toss at 48.9%.  The expected out of sample error is 51.1%!

  Random Forest
  modelFit <- train(classe ~., data=training, method="rf")
  predictions <- predict(modelFit, newdata=validation)
  confusionMatrix(validation$classe, predictions)

            Reference
  Prediction    A    B    C    D    E
           A 1674    0    0    0    0
           B   12 1126    1    0    0
           C    0    4 1018    4    0
           D    0    0    6  957    1
           E    0    0    2    4 1076

  Overall Statistics

                 Accuracy : 0.9942         
                   95% CI : (0.9919, 0.996)
      No Information Rate : 0.2865         
      P-Value [Acc > NIR] : < 2.2e-16      

                    Kappa : 0.9927         
  Mcnemar's Test P-Value : NA             
The Random Forest was very slow, three hours and forty minutes, but the prediction accuracy was very high at 99.42%.  The expected out of sample error is 0.58%. 

	Boosting
  modelFit3 <- train(classe ~., data=training, method="gbm")
  predictions3 <- predict(modelFit3, newdata=validation)
  confusionMatrix(validation$classe, predictions3)

            Reference
  Prediction    A    B    C    D    E
           A 1650   14    4    4    2
           B   36 1074   22    5    2
           C    0   32  980   11    3
           D    0    7   17  933    7
           E    1   11   12   15 1043

  Overall Statistics

                 Accuracy : 0.9652          
                   95% CI : (0.9602, 0.9697)
      No Information Rate : 0.2867          
      P-Value [Acc > NIR] : < 2.2e-16       

                    Kappa : 0.9559          
  Mcnemar's Test P-Value : 8.338e-05   

Boosting was a little less accurate than Random Forest, at 96.5%, but far faster, completing in a few minutes.  If speed became an issue, you could give up some accuracy for significant savings in run time.  The expected out of sample error is 3.48%.

Final Predictions
Apply the trained Random Forest model to our testing data to predict the new subjects’ exercise performance categories.
final <- predict(modelFit, newdata=testfileSub3)
final
    
[1] B A B A A E D B A A B C B A E E A B B B
Levels: A B C D E

