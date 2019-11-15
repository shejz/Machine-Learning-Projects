
library(data.table)
library(dplyr)
library(ggplot2)
library(stringr)
library(DT)
library(tidyr)
library(corrplot)
library(leaflet)
library(lubridate)

properties<-read.csv('https://s3.amazonaws.com/hackerday.datascience/82/properties_2016.csv',header=T,sep=',')
transactions<-read.csv('https://s3.amazonaws.com/hackerday.datascience/82/train_2016.csv',header=T,sep=',')
summary(properties)

properties <- properties %>% 
  mutate(taxdelinquencyflag = ifelse(taxdelinquencyflag=='Y',1,0),
         fireplaceflag= ifelse(fireplaceflag=='Y',1,0),
         hashottuborspa=ifelse(hashottuborspa=='Y',1,0)
         )

# Which particular month contibutes maximum number of transactions
# distribution of transactions over the length of the time period we have

Num_tr_by_M <- transactions %>%
  mutate(year_month=make_date(year=year(transactiondate),
                         month=month(transactiondate)))

Num_tr_by_M %>% 
  group_by(year_month) %>% count() %>%
  ggplot(aes(x=year_month,y=n)) +
  geom_bar(stat = 'identity',fill='red')
  
# try to understand the distribution of the target variable
transactions %>%
  ggplot(aes(x=logerror)) +
  geom_histogram(bins = 1000,fill='blue') +
  theme_bw() + ylab('Count')

#absolute value of the logerror
transactions<-transactions %>%
  mutate(abs_logerror=abs(logerror))

transactions %>%
  ggplot(aes(x=abs_logerror)) +
  geom_histogram(bins = 500,fill='blue') +
  theme_bw() + ylab('Count')

#Is there any variation(trend/over time) in the absolute value of error?
transactions %>%
  mutate(year_month=make_date(year=year(transactiondate),
                              month=month(transactiondate))) %>%
  group_by(year_month) %>% 
  summarise(mean_abs_logerror=mean(abs_logerror)) %>%
  ggplot(aes(x=year_month,y=mean_abs_logerror)) +
  geom_line(size=1.2,fill='red') +
    geom_point(size=4,color='blue')

#Is there any variation(trend/over time) in the log error?
transactions %>%
  mutate(year_month=make_date(year=year(transactiondate),
                              month=month(transactiondate))) %>%
  group_by(year_month) %>% 
  summarise(mean_logerror=mean(logerror)) %>%
  ggplot(aes(x=year_month,y=mean_logerror)) +
  geom_line(size=1.2,fill='red') +
  geom_point(size=4,color='red')

# How many missing values are there for each feature ?
# How many feauteres are there with zero missing values?
missing_values <- properties %>%
  summarise_all(funs(sum(is.na(.))/n()))

missing_values <- gather(missing_values,key='feature',value = 'missing_percentage')
missing_values %>%
  ggplot(aes(x=reorder(feature,-missing_percentage),y=missing_percentage)) +
  geom_bar(stat = 'identity',fill='orange') +
  coord_flip()
  
# finding out relevant/good features where the missing % < 15%
good_features <- filter(missing_values,missing_percentage<0.85)

#understanding the relationship between good features and the target variable
# join the two data frames
vars <- good_features$feature %>% as.numeric()

c_join <- transactions %>%
  left_join(properties,by='parcelid')

tab<- c_join %>% select(one_of(c(vars,'abs_logerror')))

corrplot(cor(tab,use='complete.obs'),type = 'lower')

# understanding the shape of the distribution
properties %>% ggplot(aes(x=yearbuilt)) + geom_line(stat = 'density',color='red')

c_join %>% 
  group_by(yearbuilt) %>%
  summarize(mean_abs_logerror=mean(abs(logerror)),n()) %>%
  ggplot(aes(x=yearbuilt,y=mean_abs_logerror)) + 
  geom_point(color='red')

###where the zestimate variable predicts well
transactions <- transactions %>% 
  mutate(percentile = cut(abs_logerror,quantile(abs_logerror,
                                                probs=c(0,0.1,0.25,0.75,0.9,1),names=F),
                          include.lowest=T,labels=F))

####Model Implementation #########
prop<-read.csv('https://s3.amazonaws.com/hackerday.datascience/82/properties_2016.csv',header=T,sep=',')

train<-read.csv('https://s3.amazonaws.com/hackerday.datascience/82/train_2016.csv',header=T,sep=',')

# feature enginerring
prop$hashottuborspa<-ifelse(prop$hashottuborspa=='true',1,0)
prop$fireplaceflag<-ifelse(prop$fireplaceflag=='true',1,0)
prop$taxdelinquencyflag<-ifelse(prop$taxdelinquencyflag=='Y',1,0)
prop$propertycountylandusecode<-as.numeric(prop$propertycountylandusecode)
prop$propertyzoningdesc<-as.numeric(prop$propertyzoningdesc)

prop<-data.table(prop)
train<-data.table(train)

setkey(prop,parcelid)
setkey(train,parcelid)

training <-prop[train]

target <-training$logerror

dtrain <-training[,!c('logerror','parcelid','transactiondate'),with=F]

feature_names <- names(dtrain)

#XGBoost set up
library(data.table)
library(caret)
library(xgboost)

dtest <- xgb.DMatrix(data=as.matrix(prop[,..feature_names]))
dtrain <- xgb.DMatrix(data=as.matrix(dtrain),label=target)

# cross validation scheme to avoid model overfitting

foldsCV <- createFolds(target,k=10,list = T,returnTrain = F)

# hyperparameter tuning
param <- list(
  objective='reg:linear',
  booster='gbtree',
  eval_metric='mae',
  eta=0.005,
  max_depth=4,
  min_child_weight=4,
  subsample=0.5,
  colsample_bytree=0.5,
  gamma=0.01
)

# train a simple model
# this function should be levergaed for hyperparameter optimization (gridExtra)

xgb_mod <- xgb.train(data=dtrain,
                     params=param,
                     nrounds = 2500,
                     print_every_n = 5)

# view the model results
# feature importance view
importance_matrix <-xgb.importance(feature_names,model=xgb_mod)
xgb.plot.importance(importance_matrix)

#predict using the test dataset
preds <- predict(xgb_mod,dtest)

#xgb_mod$params$eval_metric

results <- data.table(parcelid=prop$parcelid,
                      '201610'=preds,
                      '201611'=preds,
                      '201612'=preds,
                      '201710'=preds,
                      '201711'=preds,
                      '201712'=preds)
head(results)

# cross validation usage to get best hyper parameter
xgb_cv <- xgb.cv(data=dtrain,
                 params = param,
                 nrounds = 2500,
                 prediction = T,
                 maximize = F,
                 folds = foldsCV
                 )

#to get the best results
print(xgb_cv$evaluation_log[which.min(xgb_cv$evaluation_log$train_mae_mean)])
