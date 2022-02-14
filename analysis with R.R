library(dplyr)
library(tidyr)
library(ggplot2)
library(caTools)
library(normalr)
library(aod)
library(caret)
library(fullROC)
# 1. Import the datasets and libraries, check datatype, statistical summary, shape, null values or incorrect imputation
#the data set contains 5000 rows with 14 variables, 2 variables (ID, zip.code) are not used to build model
#5 variables (Personal.Loan, Securities.Account, CD.Account, Online, CreditCard) are binary variables with only (0,1) values
data <- read.table("Dataset.csv", header=TRUE, sep = ",", stringsAsFactors = FALSE)
typeof(data)
summary(data)
dim(data)
sapply( list(1,NULL,3), function(x) length(data) == 0 )

# 2(1). Number of unique in each column
data %>% summarise_all(n_distinct)

# 2(2). Number of people with zero mortgage
#in total 3462 people out of 5000 have zero mortgage with the bank
length(which(data$Mortgage == "0"))

# 2(3). Number of people with zero credit card spending per month
#in total 3530 people out of 5000 have zero credit card spending per month in this bank
length(which(data$CreditCard == "0"))

# 2(4). Value counts of all categorical columns
#value count of Personal.Loan:4520 data points with 0, 480 data points with 1
#Securities.Account:4478 data points with 0, 522 data points with 1
#CD.Account:4698 data points with 0, 302 data points with 1
#Online:2016 data points with 0, 2984 data points with 1
#CreditCard:3530 data points with 0, 1470 data points with 1
value_counts<-function(x){
  table(x)
}
value_counts(data$CreditCard)

# 2(5). Univariate and Bivariate
#Univariate analysis
uni<-function(x){
  summary(x)
}
uni(data$Age)
uni(data$CreditCard)

ggplot(data, aes(x = `Age`)) +
  geom_bar() +
  xlab("Age") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#Bivariate analysis
#age vs mortgage: graphically people around 35 or between 45-50 have more mortgage than other age groups
age_mortgage <- ggplot(data, aes(Age, Mortgage)) + geom_point() +
      labs(title = "Distribution of mortgage relative to age", x = "age", y = "mortgage amount") +
      theme_minimal()

#income vs mortgage: graphically people with higher income also have higher amount of mortgage
income_mortgage <- ggplot(data, aes(Income, Mortgage)) + geom_point() +
  labs(title = "Distribution of mortgage relative to income", x = "income", y = "mortgage amount") +
  theme_minimal()

#income with personal loan: graphically people with higher income also have higher chance to have personal loan
income_loan <- ggplot(data, aes(Income, Personal.Loan)) + geom_point() +
  labs(title = "Distribution of mortgage relative to Personal.Loan", x = "income", y = "Personal.Loan") +
  theme_minimal()

# 3. Split the data into training and test set in the ratio of 70:30 respectively
dat<-normalise(data)
data_split <-createDataPartition(y=dat$Personal.Loan,p=0.7,list=FALSE)
train<-data[data_split,]
test<-data[-data_split,]

# 4. Use the Logistic Regression model to predict whether the customer will take a personal loan or not.
#Print all the metrics related to evaluating the model performance (accuracy, recall, precision, f1score, and roc_auc_score).
#Draw a heatmap to display confusion matrix

#Training the model with GLM
#by looking at the summary of train_model, the deviance of null is 2172 compare to 3499 degrees of freedom
#the deviance of residual is 2170.4 compare to 3488 degrees of freedom
#this model is relatively a good fit
train_model<-train(train[, c(2,3,4,6,7,8,9,11,12,13,14)],factor(train[, 10]), method='glm')

# Predict the labels of the test set
predictions<-predict(object=train_model,newdata=data.frame(test[, c(2,3,4,6,7,8,9,11,12,13,14)]),type="raw")
# Evaluate the predictions
table(predictions)
# Construct confusion matrix
a<-factor(predictions, levels=min(test$Personal.Loan):max(test$Personal.Loan))
b<-factor(test$Personal.Loan, levels=min(test$Personal.Loan):max(test$Personal.Loan))
result<-confusionMatrix(a,b)
#Accuracy, recall, precision, f1score, and roc_auc_score
accuracy<-result$overall['Accuracy']
precision<-result$byClass['Pos Pred Value'] 
recall<-result$byClass['Sensitivity']
f_measure <- 2 * ((precision * recall) / (precision + recall))
roc_auc<-0.5*((1335/(1335+28))+(90/(90+47)))
list(accuracy,precision,recall,f_measure,roc_auc)

#heatmap
#True positive has 1335 value counts, True negative has 90 value counts, false positive has 47 value counts, false negative has 28 value counts.
#convert the confusion matrix to a numeric matrix:0 is the positive in this case, so TRUE POSITIVE is (0,0)
#we can see from the heatmap, the true positive(0,0)has the highest value which 1335, next is true negative(1,1) which is 90.
Prediction <- factor(c(0, 0, 1, 1))
Reference <- factor(c(1, 0, 1, 0))
Y <- c(28,1335,90,47)
df <- data.frame(Reference, Prediction, Y)
ggplot(data = df, mapping = aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Y), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
  scale_fill_gradient(low = "light blue", high = "dark blue") +
  theme_bw() + theme(legend.position = "none")

# 5. Find out coefficients of all the attributes and show the output in a data frame with column names
coefficients_of_the_attributes<-list(train_model$finalModel)

#For test data show all the rows where the predicted class is not equal to the observed class.
observed_class<-test$Personal.Loan
predicted_class<-predictions
rows_where_different<-which(observed_class!=predicted_class)

# 6. Give conclusion related to the Business understanding of your model:
#Based on the confusion matrix from the training model, this model actually has an accuracy of 0.95, precision of 0.965, and Roc_auc_score is 0.81, so we can conclude that this model is relatively a good fit.
#Using the analysis result from the fitted model, we can see that over the 11 factors displayed, CD.Account, Education, Family, CCAvg, Income have the highest impact for Personal.Loan.
#So in the future, if the marketing department want to find out which customers will most likely apply for the loan, they should target customers with the following characteristics:
#A liability customer(CD.Account) with high Education, bigger family size, high average spending on credit card(CCAvg), and high income.
#In addition, if we can have credit card spending amount and CD account amount, those data will be more valuable for fitting the model.
