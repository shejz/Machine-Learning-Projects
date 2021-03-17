library(h2o)
h2o.init(nthreads = -1)

# This step takes a few seconds bc we have to download the data from the internet...
train_file <- "https://h2o-public-test-data.s3.amazonaws.com/bigdata/laptop/mnist/train.csv.gz"
test_file <- "https://h2o-public-test-data.s3.amazonaws.com/bigdata/laptop/mnist/test.csv.gz"
train <- h2o.importFile(train_file)
test <- h2o.importFile(test_file)

y <- "C785"  #response column: digits 0-9
x <- setdiff(names(train), y)  #vector of predictor column names

# Since the response is encoded as integers, we need to tell H2O that
# the response is in fact a categorical/factor column.  Otherwise, it 
# will train a regression model instead of multiclass classification.
train[,y] <- as.factor(train[,y])
test[,y] <- as.factor(test[,y])

dl_fit1 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "dl_fit1",
                            hidden = c(20,20),
                            seed = 1)

dl_fit1
#######################################################
dl_fit2 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "dl_fit2",
                            epochs = 50,
                            hidden = c(20,20),
                            stopping_rounds = 0,  # disable early stopping
                            seed = 1)
dl_fit2
######################################################

dl_fit3 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "dl_fit3",
                            epochs = 50,
                            hidden = c(50,30,10),
                            nfolds = 3,                            #used for early stopping
                            score_interval = 1,                    #used for early stopping
                            stopping_rounds = 5,                   #used for early stopping
                            stopping_metric = "misclassification", #used for early stopping
                            stopping_tolerance = 1e-3,             #used for early stopping
                            seed = 1)
dl_fit3
#################################
dl_perf1 <- h2o.performance(model = dl_fit1, newdata = test)
dl_perf2 <- h2o.performance(model = dl_fit2, newdata = test)
dl_perf3 <- h2o.performance(model = dl_fit3, newdata = test)

# Retreive test set MSE
h2o.mse(dl_perf1)

h2o.mse(dl_perf2) 

h2o.mse(dl_perf3)

h2o.scoreHistory(dl_fit3)

h2o.confusionMatrix(dl_fit3)

plot(dl_fit3, 
     timestep = "epochs", 
     metric = "classification_error")


# Get the CV models from the `dl_fit3` object
cv_models <- sapply(dl_fit3@model$cross_validation_models, 
                    function(i) h2o.getModel(i$name))

# Plot the scoring history over time
plot(cv_models[[1]], 
     timestep = "epochs", 
     metric = "classification_error")

#########################Grid Search##########
activation_opt <- c("Rectifier", "Maxout", "Tanh")
l1_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01)
l2_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01)

hyper_params <- list(activation = activation_opt, l1 = l1_opt, l2 = l2_opt)
search_criteria <- list(strategy = "RandomDiscrete", max_runtime_secs = 60)


splits <- h2o.splitFrame(train, ratios = 0.8, seed = 1)

dl_grid <- h2o.grid("deeplearning", x = x, y = y,
                    grid_id = "dl_grid",
                    training_frame = splits[[1]],
                    validation_frame = splits[[2]],
                    seed = 1,
                    hidden = c(20,20),
                    hyper_params = hyper_params,
                    search_criteria = search_criteria)


dl_gridperf <- h2o.getGrid(grid_id = "dl_grid", 
                           sort_by = "accuracy", 
                           decreasing = TRUE)
print(dl_gridperf)

best_dl_model_id <- dl_gridperf@model_ids[[1]]
best_dl <- h2o.getModel(best_dl_model_id)


best_dl_perf <- h2o.performance(model = best_dl, newdata = test)
h2o.mse(best_dl_perf)


train_file <- "https://h2o-public-test-data.s3.amazonaws.com/bigdata/laptop/mnist/train.csv.gz"
test_file <- "https://h2o-public-test-data.s3.amazonaws.com/bigdata/laptop/mnist/test.csv.gz"
train <- h2o.importFile(train_file)
test <- h2o.importFile(test_file)
y <- "C785"  #response column: digits 0-9
x <- setdiff(names(train), y)  #vector of predictor column names
train[,y] <- as.factor(train[,y])
test[,y] <- as.factor(test[,y])

splits <- h2o.splitFrame(train, 0.5, seed = 1)

# first part of the data, without labels for unsupervised learning
train_unsupervised <- splits[[1]]

# second part of the data, with labels for supervised learning
train_supervised <- splits[[2]]

dim(train_supervised)

dim(train_unsupervised)

hidden <- c(128, 64, 128)

ae_model <- h2o.deeplearning(x = x, 
                             training_frame = train_unsupervised,
                             model_id = "mnist_autoencoder",
                             ignore_const_cols = FALSE,
                             activation = "Tanh",  # Tanh is good for autoencoding
                             hidden = hidden,
                             autoencoder = TRUE)

fit1 <- h2o.deeplearning(x = x, y = y,
                         training_frame = train_supervised,
                         ignore_const_cols = FALSE,
                         hidden = hidden,
                         pretrained_autoencoder = "mnist_autoencoder")

perf1 <- h2o.performance(fit1, newdata = test)
h2o.mse(perf1)

fit2 <- h2o.deeplearning(x = x, y = y,
                         training_frame = train_supervised,
                         ignore_const_cols = FALSE,
                         hidden = hidden)

perf2 <- h2o.performance(fit2, newdata = test)
h2o.mse(perf2)

# convert train_supervised with autoencoder model to lower-dimensional space
train_reduced_x <- h2o.deepfeatures(ae_model, train_supervised, layer = 1)
dim(train_reduced_x)

# Now train DRF on reduced feature space, first need to add response back
train_reduced <- h2o.cbind(train_reduced_x, train_supervised[,y])

rf1 <- h2o.randomForest(x = names(train_reduced_x), y = y, 
                        training_frame = train_reduced,
                        ntrees = 100, seed = 1)

test_reduced_x <- h2o.deepfeatures(ae_model, test, layer = 1)
test_reduced <- h2o.cbind(test_reduced_x, test[,y])

rf_perf <- h2o.performance(rf1, newdata = test_reduced)
h2o.mse(rf_perf)


test_rec_error <- as.data.frame(h2o.anomaly(ae_model, test)) 

test_recon <- predict(ae_model, test)
