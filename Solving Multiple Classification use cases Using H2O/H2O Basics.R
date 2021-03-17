#The dataset is then assigned an identifier (the .hex file type in H2O) used as a reference in 
#commands to the web server
library(h2o)
# Starts H2O using localhost IP, port 54321, all CPUs,and 4g of memory
h2o.init(ip = "localhost", port = 54321, nthreads= -1,max_mem_size = "4g")

h2o.init()

h2o.clusterInfo()

library(h2o)
h2o.init()
airlinesURL = "https://s3.amazonaws.com/h2o-airlines-unpacked/allyears2k.csv"
airlines.hex = h2o.importFile(path = airlinesURL,
                                destination_frame = "airlines.hex")
summary(airlines.hex)

# View quantiles and histograms
quantile(x = airlines.hex$ArrDelay, na.rm = TRUE)
h2o.hist(airlines.hex$ArrDelay)

# Find number of flights by airport
originFlights = h2o.group_by(data = airlines.hex, by = "Origin", nrow("Origin"),
                             gb.control=list(na.methods="rm"))
originFlights.R = as.data.frame(originFlights)

# Find number of flights per month
flightsByMonth = h2o.group_by(data = airlines.hex, by= "Month", nrow("Month"),
                              gb.control=list(na.methods="rm"))
flightsByMonth.R = as.data.frame(flightsByMonth)

#Find months with the highest cancellation ratio
which(colnames(airlines.hex)=="Cancelled")
cancellationsByMonth = h2o.group_by(data = airlines.hex, by = "Month", sum("Cancelled"),gb.control=
                                         list(na.methods="rm"))
cancellation_rate = cancellationsByMonth$sum_Cancelled/flightsByMonth$nrow_Month
rates_table = h2o.cbind(flightsByMonth$Month,
                           cancellation_rate)
rates_table.R = as.data.frame(rates_table)

# Construct test and train sets using sampling
airlines.split = h2o.splitFrame(data = airlines.hex, ratios = 0.85)
airlines.train = airlines.split[[1]]
airlines.test = airlines.split[[2]]

# Display a summary using table-like functions
h2o.table(airlines.train$Cancelled)
h2o.table(airlines.test$Cancelled)

# Set predictor and response variables
Y = "IsDepDelayed"
X = c("Origin", "Dest", "DayofMonth", "Year", "
         UniqueCarrier", "DayOfWeek", "Month", "DepTime", "
         ArrTime", "Distance")

# Define the data for the model and display the results
airlines.glm <- h2o.glm(training_frame=airlines.train, y="IsDepDelayed", x=c("Origin", "Dest", "DayofMonth", "Year", "UniqueCarrier", 
"DayOfWeek", "Month", "DepTime", "ArrTime", "Distance"), family = "binomial", alpha = 0.5)

# View model information: training statistics, performance, important variables
summary(airlines.glm)

# Predict using GLM model
pred = h2o.predict(object = airlines.glm, newdata =airlines.test)

# Look at summary of predictions: probability of TRUE class (p1)
summary(pred$predict)

######################
data(iris)
iris.hex <- as.h2o(iris,destination_frame = "iris.hex")
iris.gbm <- h2o.gbm(y = 1, x = 2:5, training_frame =iris.hex, ntrees = 10,
                    max_depth = 3,min_rows = 2, learn_rate = 0.2,
                        distribution= "gaussian")

# To obtain the Mean-squared Error by tree from the
#model object:
iris.gbm@model$scoring_history


#To generate a classification model that uses labels, use distribution= "multinomial"
iris.gbm2 <- h2o.gbm(y = 5, x = 1:4, training_frame = iris.hex, ntrees = 15, max_depth = 5, min_rows = 2, learn_rate = 0.01, distribution= "multinomial")
iris.gbm2@model$training_metrics

################################
prostate.hex <- h2o.importFile(path = "https://raw.github.com/h2oai/h2o/master/smalldata/logreg/prostate.csv" , destination_frame = "prostate.hex" )
prostate.glm<-h2o.glm(y = "CAPSULE", x = c("AGE","RACE","PSA","DCAPS"), training_frame = prostate.hex,family = "binomial", nfolds = 10, alpha =0.5)
prostate.glm@model$cross_validation_metrics

########
h2o.kmeans(training_frame = iris.hex, k = 3, x = 1:4)

########
ausPath = system.file("extdata", "australia.csv", package="h2o")
australia.hex = h2o.importFile(path = ausPath)
australia.pca <- h2o.prcomp(training_frame =
                                  australia.hex, transform = "STANDARDIZE",k = 3)
australia.pca

##############
prostate.fit = h2o.predict(object = prostate.glm,newdata = prostate.hex)
prostate.fit
