# Libraries
library(caTools)
library(moments)
library(e1071)
library(corrplot)
library(caret)
library(cluster)
library(rpart)
library(rpart.plot)
library(caret)
library(dendextend)
library(neuralnet)
library(randomForest)
library(adabag)

# Logistic Regression
dataset <- read.csv('framingham.csv')
View(dataset)

dim(dataset)
str(dataset)
table(dataset$TenYearCHD)

# Look for Outliers
boxplot(dataset$education)
boxplot(dataset$cigsPerDay)
boxplot(dataset$BPMeds)
boxplot(dataset$totChol)
boxplot(dataset$BMI)
boxplot(dataset$heartRate)
boxplot(dataset$glucose)

boxplot(dataset, col = rainbow(ncol(dataset)))

# Histogram
ggplot(dataset, aes(x=cigsPerDay)) + 
  geom_histogram(binwidth=5, position="identity", fill="#69b3a2", color="#e9ecef", alpha=0.9)

ggplot(dataset, aes(x=totChol)) +
  geom_histogram(binwidth=20, position="identity", fill="#69b3a2", color="#e9ecef", alpha=0.9)

ggplot(dataset, aes(x=heartRate)) +
  geom_histogram(binwidth=5, position="identity", fill="#69b3a2", color="#e9ecef", alpha=0.9)

range(dataset$totChol)

# Skewness
skewness(dataset)

# Find the features with NAs
colSums(is.na(dataset))
sum(is.na(dataset))

# Imputing missing values
education.median <- median(dataset$education, na.rm = TRUE)
dataset[is.na(dataset$education) == TRUE, "education"] <- education.median

cigsPerDay.median <- median(dataset$cigsPerDay, na.rm = TRUE)
dataset[is.na(dataset$cigsPerDay) == TRUE, "cigsPerDay"] <- cigsPerDay.median

BPMeds.null <- 0
dataset[is.na(dataset$BPMeds) == TRUE, "BPMeds"] <- BPMeds.null

totChol.median <- median(dataset$totChol, na.rm = TRUE)
dataset[is.na(dataset$totChol) == TRUE, "totChol"] <- totChol.median

BMI.median <- median(dataset$BMI, na.rm = TRUE)
dataset[is.na(dataset$BMI)== TRUE, "BMI"] <- BMI.median

heartRate.median <- median(dataset$heartRate, na.rm = TRUE)
dataset[is.na(dataset$heartRate) == TRUE, "heartRate"] <- heartRate.median

glucose.median <- median(dataset$glucose, na.rm = TRUE)
dataset[is.na(dataset$glucose) == TRUE, "glucose"] <- glucose.median

# Check for NAs after imputing  
colSums(is.na(dataset))
sum(is.na(dataset))

# Understand the summary and the corelation of the features.
summary(dataset)
Correlation = cor(dataset, use="complete.obs")
corrplot(Correlation, method="number",main="Correlation Matrix", mar=c(0,0,1,0), tl.cex=0.8, tl.col="black", tl.srt=45)

# Plot age and sysBP to understand the relation. 
sysBP.mean <- mean(dataset$sysBP, na.rm = TRUE)
plot(dataset$age, dataset$sysBP)
points(dataset$age, dataset$sysBP, col = ifelse(dataset$male == 0, "blue", "red"), pch = 19)
abline(h = sysBP.mean, col = "black")

# Split the dataset into test and train 
split <- sample.split(dataset$TenYearCHD, SplitRatio = 0.7)
train <- subset(dataset, split == TRUE)
test <- subset(dataset, split == FALSE)

# Logistic Model
logit_model <-  glm(TenYearCHD ~ . , train, family = "binomial")
options(scipen=999)
summary(logit_model)

# Training data Confusion Matrix
log.pred.train = predict(logit_model, train, type = "response")
log.pred.train = ifelse(log.pred.train > 0.5, 1, 0)
confusionMatrix(as.factor(log.pred.train), as.factor(train$TenYearCHD))

# Test data Confusion Matrix
test_conf <- confusionMatrix(factor(test_pred_CHD), factor(test$TenYearCHD))
test_conf

# use predict() with type = "response" to compute predicted probabilities. 
logit.reg.pred <- predict(logit_model, test, type = "response")

# first 5 actual and predicted records
data.frame(actual = test$TenYearCHD[1:5], predicted = logit.reg.pred[1:5])

logit.reg.pred.classes <- ifelse(logit.reg.pred > 0.6, 1, 0)
confusionMatrix(as.factor(logit.reg.pred.classes), as.factor(test$TenYearCHD))

# model selection
full.logit.reg <- glm(TenYearCHD ~ ., data = train, family = "binomial") 
empty.logit.reg  <- glm(TenYearCHD ~ 1,data = train, family= "binomial")
summary(empty.logit.reg)

backwards = step(full.logit.reg) # Smaller the AIC, better the model performance
summary(backwards)
formula(backwards)

backwards.reg.pred <- predict(backwards, test, type = "response")
backwards.reg.pred.classes <- ifelse(backwards.reg.pred > 0.6, 1, 0)
confusionMatrix(as.factor(backwards.reg.pred.classes), as.factor(test$TenYearCHD))

back2 <- glm(TenYearCHD ~ .,data = train, family= "binomial")
summary(back2)

forwards = step(empty.logit.reg,scope=list(lower=formula(empty.logit.reg),upper=formula(full.logit.reg)), direction="forward",trace=1)
formula(forwards)

stepwise = step(empty.logit.reg,scope=list(lower=formula(empty.logit.reg),upper=formula(full.logit.reg)), direction="both",trace=1)
formula(stepwise)

#Neural Networks
set.seed(42)
nn <- neuralnet(TenYearCHD ~ ., data = train, act.fct = "logistic", linear.output = F, hidden = 3)
plot(nn, rep="best")

nn.pred <- predict(nn, test, type = "response")
nn.pred.classes <- ifelse(nn.pred > 0.6, 1, 0)
confusionMatrix(as.factor(nn.pred.classes), as.factor(test$TenYearCHD))

#Decision Tree
dataset.df <- read.csv("framingham.csv")

View(dataset.df)
# Splitting data into train and validation
set.seed(42)  
train.index <- sample(c(1:dim(dataset.df)[1]), dim(dataset.df)[1]*0.7)  
train.df <- dataset.df[train.index, ]
valid.df <- dataset.df[-train.index, ]

# Classification tree
default.ct <- rpart(TenYearCHD ~ ., data = train.df ,method = "class")

# Plot tree
prp(default.ct, type = 1, extra = 2, under = TRUE, split.font = 1, varlen = -10)
# Count number of leaves
length(default.ct$frame$var[default.ct$frame$var == "<leaf>"])

default.info.ct <- rpart(TenYearCHD ~ ., data = train.df, parms = list(split = 'information'), method = "class")
prp(default.info.ct, type = 1, extra = 2, under = TRUE, split.font = 1, varlen = -10)
length(default.info.ct$frame$var[default.info.ct$frame$var == "<leaf>"])

deeper.ct <- rpart(TenYearCHD ~ ., data = train.df, method = "class", cp = -1, minsplit = 1)
length(deeper.ct$frame$var[deeper.ct$frame$var == "<leaf>"])
prp(deeper.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, 
    box.col=ifelse(deeper.ct$frame$var == "<leaf>", 'gray', 'white'))  

# Classify records in the validation data.
# Set argument type = "class" in predict() to generate predicted class membership.
default.ct.point.pred.train <- predict(default.ct,train.df,type = "class")

# Generate confusion matrix for training data
confusionMatrix(default.ct.point.pred.train, as.factor(train.df$TenYearCHD))

# Repeat the code for the validation set, and the deeper tree
default.ct.point.pred.valid <- predict(default.ct,valid.df,type = "class")
confusionMatrix(default.ct.point.pred.valid, as.factor(valid.df$TenYearCHD))

deeper.ct.point.pred.train <- predict(deeper.ct,train.df,type = "class")
confusionMatrix(deeper.ct.point.pred.train, as.factor(train.df$TenYearCHD))
deeper.ct.point.pred.valid <- predict(deeper.ct,valid.df,type = "class")
confusionMatrix(deeper.ct.point.pred.valid, as.factor(valid.df$TenYearCHD))

set.seed(1)
cv.ct <- rpart(TenYearCHD ~ ., data = train.df, method = "class", cp = 0.00001, minsplit = 1, xval = 5)  # minsplit is the minimum number of observations in a node for a split to be attempted. xval is number K of folds in a K-fold cross-validation.
printcp(cv.ct)  # Print out the cp table of cross-validation errors. The R-squared for a regression tree is 1 minus rel error. xerror (or relative cross-validation error where "x" stands for "cross") is a scaled version of overall average of the 5 out-of-sample errors across the 5 folds.
pruned.ct <- prune(cv.ct, cp = 0.0068729)

printcp(pruned.ct)
prp(pruned.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, 
    box.col=ifelse(pruned.ct$frame$var == "<leaf>", 'gray', 'white')) 


# Random Forest
rf <- randomForest(as.factor(TenYearCHD) ~ ., data = train.df, ntree = 500, 
                   mtry = 4, nodesize = 5, importance = TRUE, na.action=na.exclude)  

# Variable importance plot
varImpPlot(rf, type = 1)

# Confusion matrix
rf.pred <- predict(rf, valid.df)
confusionMatrix(rf.pred, as.factor(valid.df$TenYearCHD))

# Boosted Tree
train.df$TenYearCHD <- as.factor(train.df$TenYearCHD)

set.seed(1)
boost <- boosting(TenYearCHD ~ ., data = train.df)
pred <- predict(boost, valid.df)
confusionMatrix(as.factor(pred$class), as.factor(valid.df$TenYearCHD))

# Principal Component Analysis
pcs <- prcomp(dataset)
summary(pcs) 
pcs$rot
scores <- pcs$x
head(scores, 5)