#### Loan Approval prediction

details <- read.csv("train_data.csv",na.strings= c("", "NA"), header=TRUE)
test_data <- read.csv("test_data.csv",na.strings= c("", "NA"), header=TRUE)

summary(details)
str(details)
library(plyr)
details$Dependents <- revalue(details$Dependents, c("3+"="3"))
sum(is.na(details$Dependents))
library(Amelia)
missmap(details, Main = 'Missmap', col = c("Yellow", "Black"), legends = FALSE)


#Converting these data to factor
details$Gender <- factor(details$Gender, levels = c('Female','Male'), labels = c(0,1))
details$Married <- factor(details$Married, levels = c('Yes','No'), labels = c(0,1))
details$Education <- factor(details$Education, levels = c('Graduate','Not Graduate'), labels = c(0,1))
details$Self_Employed <- factor(details$Self_Employed, levels = c('No','Yes'), labels = c(0,1))
details$Property_Area <- factor(details$Property_Area, levels = c('Rural','Semiurban', 'Urban'), labels = c(0,1,2))
details$Dependents <- factor(details$Dependents, levels =  c('0','1','2','3+'), labels = c(0,1,2,3))
details$Credit_History <- factor(details$Credit_History, levels = c("0","1"), labels = c(0,1))
details$Loan_Status <- factor(details$Loan_Status, levels = c('N','Y'), labels = c(0,1))


#test_set 

test_data$Gender <- factor(test_data$Gender, levels = c('Female','Male'), labels = c(0,1))
test_data$Married <- factor(test_data$Married, levels = c('Yes','No'), labels = c(0,1))
test_data$Education <- factor(test_data$Education, levels = c('Graduate','Not Graduate'), labels = c(0,1))
test_data$Self_Employed <- factor(test_data$Self_Employed, levels = c('No','Yes'), labels = c(0,1))
test_data$Property_Area <- factor(test_data$Property_Area, levels = c('Rural','Semiurban', 'Urban'), labels = c(0,1,2))
test_data$Dependents <- factor(test_data$Dependents, levels =  c('0','1','2','3'), labels = c(0,1,2,3))
test_data$Credit_History <- factor(test_data$Credit_History, levels = c("0","1"), labels = c(0,1))


#dealing with missing data of LoanAmount, Loan_Amount_Term
details$LoanAmount <- ifelse(is.na(details$LoanAmount),
                            ave(details$LoanAmount, FUN = function(x)mean(x, na.rm = TRUE
                            )),
                            details$LoanAmount)

details$Loan_Amount_Term <- ifelse(is.na(details$Loan_Amount_Term),
                                  ave(details$Loan_Amount_Term, FUN = function(x)mean(x, na.rm = TRUE
                                  )),
                                  details$Loan_Amount_Term)

test_data$LoanAmount <- ifelse(is.na(test_data$LoanAmount),
                             ave(test_data$LoanAmount, FUN = function(x)mean(x, na.rm = TRUE
                             )),
                             test_data$LoanAmount)

test_data$Loan_Amount_Term <- ifelse(is.na(test_data$Loan_Amount_Term),
                                   ave(test_data$Loan_Amount_Term, FUN = function(x)mean(x, na.rm = TRUE
                                   )),
                                   test_data$Loan_Amount_Term)

#dealing with missing values in Gender, Married, Education, Self_Employed and Property_Area using median
details$Gender[is.na(details$Gender)] <- 1
details$Married[is.na(details$Married)] <- 0
details$Dependents[is.na(details$Dependents)] <- 1
details$Education[is.na(details$Education)] <- 0
details$Self_Employed[is.na(details$Self_Employed)] <- 0
details$Credit_History[is.na(details$Credit_History)] <- 1

#similarly for test_data

test_data$Gender[is.na(test_data$Gender)] <- 1
test_data$Married[is.na(test_data$Married)] <- 0
test_data$Dependents[is.na(test_data$Dependents)] <- 1
test_data$Education[is.na(test_data$Education)] <- 0
test_data$Self_Employed[is.na(test_data$Self_Employed)] <- 0
test_data$Credit_History[is.na(test_data$Credit_History)] <- 1

missmap(details, Main = 'Missmap', col = c("Yellow", "Black"), legends = FALSE)
# now we have no missing values in the data

# Data visualization

library(ggplot2)
library(gridExtra)
plot1 <- ggplot(details,aes(Loan_Status)) + geom_bar()
plot2 <- ggplot(details,aes(Education)) + geom_bar(aes(fill=factor(Education)),alpha=0.5)
plot3 <- ggplot(details,aes(Gender)) + geom_bar(aes(fill=factor(Gender)),alpha=0.5)
plot4 <- ggplot(details,aes(LoanAmount)) + geom_histogram(fill='blue',bins=20,alpha=0.5)
gridExtra::grid.arrange(plot1, plot2, plot3, plot4,ncol=2)


#Feature scaling 
details[,6:9] <- scale(details[,6:9])
test_data[, 6:9] <- scale(test_data[, 6:9])

# Splitting train data into training and testing set
library(caTools)
set.seed(51)
split <- sample.split(details$Loan_Status, SplitRatio = 0.7)
train <- subset(details[-1],split == T)
test <- subset(details[-1],split == F)

#Log Model

log.fit <- glm(formula = Loan_Status ~ .,family = binomial(link = 'logit'), 
               data = train)

summary(log.fit)
fitted.probablities_log <- predict(log.fit,test, type = 'response')
fitted.results_log <- ifelse(fitted.probablities_log > 0.5 , 1, 0)
misclasserror_log <- mean(fitted.results_log != test$Loan_Status)
print(1- misclasserror_log)
#  Accurracy = 0.8162162


prob_predlog <- predict(log.fit, type = 'response', newdata = test_data)
Loan_Statuslog <- ifelse(prob_predlog > 0.5, 1, 0)

Loan_Status <- ifelse(Loan_Statuslog == 1, "Y","N")
Loan_ID <- test_data$Loan_ID
Logresult <- cbind(Loan_ID, Loan_Status)
write.csv(Logresult, file = "Finallog.csv")


#SVM model

library(e1071)
svm.fit <- svm(Loan_Status~., data = train , type = "C-classification", 
               kernel = 'linear')
summary(svm.fit)


fitted.results_svm <- predict(svm.fit,test, type = 'class')

# Confusion matrix
table(fitted.results_svm,test$Loan_Status)

misclasserror_svm <- mean(fitted.results_svm != test$Loan_Status)
print(1- misclasserror_svm)
# Accuracy = 0.8108108

prob_predsvm <- predict(svm.fit, type = 'response', newdata = test_data)

Loan_Status_svm <- ifelse(prob_predsvm == 1, "Y","N")
Svm_result <- cbind(Loan_ID, Loan_Status_svm)
write.csv(Logresult, file = "Finalsvm.csv")

#Naive Bayes
library(e1071)
naive.bayes.fit <- naiveBayes(x = train[-12],
                        y = train$Loan_Status)

summary(naive.bayes.fit)
fitted.results_bayes <- predict(naive.bayes.fit,test)

misclasserror_bayes <- mean(fitted.results_bayes != test$Loan_Status)
print(1- misclasserror_bayes)
# Accuracy = 0.8054054

y_pred_bayes <- predict(naive.bayes.fit, newdata = test_data)
Loany_bayes <- ifelse(y_pred_bayes == 1, "Y","N")
Bayes_result <- cbind(Loan_ID, Loany_bayes)
colnames(Bayes_result)
write.csv(Bayes_result, file = "FinalnaiveBayes.csv")


#Random Forest

library(randomForest)
rf <- randomForest(Loan_Status ~.,data = train,ntree = 290, mtry = 2)
rf$confusion
summary(rf)
fitted.results_rf <- predict(rf,test)

misclasserror_rf <- mean(fitted.results_rf != test$Loan_Status)
print(1- misclasserror_rf)
# Accuracy = 0.8108108

y_pred_rf <- predict(rf, newdata = test_data)
Loany_rf <- ifelse(y_pred_rf == 1, "Y","N")
Logresult_rf <- cbind(Loan_ID, Loany_rf)
write.csv(Logresult, file = "RandomTree.csv")

#ROC curves for all the models

library(plotROC)
library(ROCR)

# Logistic
pred <- prediction(fitted.results_log, test$Loan_Status)
roc = performance(pred, "tpr", "fpr")
plot(roc, col=4, main="ROC curves of different machine learning classifier")

# Draw a legend.
legend(x = 0.8, y = 0.4,
       legend=c('Logistic', 'SVM','Naive Bayes', 'Random Forest'),
       col=c(4,5,6,7),
       lwd=4, cex =0.7, xpd = TRUE, horiz = FALSE)

# SVM
pred_svm <- prediction(as.numeric(fitted.results_svm), as.numeric(test$Loan_Status))
roc_svm <- performance(pred_svm, "tpr", "fpr")
plot(roc_svm, col = 5, lwd = 2, add = TRUE)

# Naive Bayes
pred_bayes <- prediction(as.numeric(fitted.results_bayes), as.numeric(test$Loan_Status))
roc_bayes <- performance(pred_bayes, "tpr", "fpr")
plot(roc_bayes, col = 6, lwd = 2, add = TRUE)

# Random Forest
pred_rf <- prediction(as.numeric(fitted.results_rf), as.numeric(test$Loan_Status))
roc_rf <- performance(pred_rf, "tpr", "fpr")
plot(roc_rf, col = 7, lwd = 2, add = TRUE)

