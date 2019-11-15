
data <- read.csv("dataset/creditcard.csv")

# Choose file
data <- read.csv(file.choose())
str(data)

# Get the Class column imbalanced data percentage
table(data$Class)/length(data$Class)*100

# EDA

# Univariate  : Distributions
# Bi-variate  : 2 Variables
# Multivariate: Multiple Variables Correlations

hist(data$Amount)
length(data$Amount[data$Amount < 300])/length(data$Amount) * 100
hist(data$Amount[data$Amount < 300])

hist(data$Time)

# Bi-variate
cor(data$Time, data$Amount)

library(corrplot)
library(caret)

corr_mat <- cor(data)
corrplot(corr_mat, method = "number")
caret::featurePlot(x=data[,2:29], y=data[,31])

# PCA
data("mtcars")
prin_comps <- princomp(mtcars)
prin_comps$loadings
prin_comps$scores

set.seed(1)
data$Class <- as.factor(data$Class)
train_index <- createDataPartition(y=data$Class, p=0.70, times=1, list=F)
train <- data[train_index,]
test <- data[-train_index,]

# Cross Validation

control <- trainControl(method = "repeatedcv", number = 5, repeats = 3)


# LDA
lda <- train(Class~., 
             data = train, 
             method="lda",
             metric="Accuracy", 
             trControl=control)

# GLM
glm <- train(Class~., 
             data = train, 
             method="glm",
             metric="Accuracy", 
             trControl=control)

# SVM
library(e1071)
svm <- train(Class~.,data = train)
results <- resampless(list(lda=lda, logistic_reg=glm))
summary(resuls)

