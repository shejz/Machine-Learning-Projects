

library(reshape)
library(reshape2)
library(xgboost)
library(caret)
library(jsonlite)
library(dplyr)
library(Matrix)
library(doParallel)
library(lubridate)




train<-read.csv(url("https://s3.amazonaws.com/hackerday.datascience/102/train.csv"))
test<-read.csv(url("https://s3.amazonaws.com/hackerday.datascience/102/test.csv"))

train$key<-paste(train$msno,train$song_id,sep="_")
test$key<-paste(test$msno,test$song_id,sep="_")

train$id<-row.names(train)

test$target<-''

train$type<-'train'
test$type<-'test'


train1<-train[,c('key','type','id','source_system_tab','source_screen_name','source_type','target')]
test1<-test[,c('key','type','id','source_system_tab','source_screen_name','source_type','target')]

master_df<-rbind(train1,test1)

rm(train)
rm(test)
rm(train1)
rm(test1)



#Creating source system tab based primary variables

master_df$flag_source_system_tab_discover<-ifelse(master_df$source_system_tab=="discover",1,0)
master_df$flag_source_system_tab_explore<-ifelse(master_df$source_system_tab=="explore",1,0)
master_df$flag_source_system_tab_listen_with<-ifelse(master_df$source_system_tab=="listen with",1,0)
master_df$flag_source_system_tab_my_library<-ifelse(master_df$source_system_tab=="my library",1,0)
master_df$flag_source_system_tab_notification<-ifelse(master_df$source_system_tab=="notification",1,0)
master_df$flag_source_system_tab_radio<-ifelse(master_df$source_system_tab=="radio",1,0)
master_df$flag_source_system_tab_search<-ifelse(master_df$source_system_tab=="search",1,0)
master_df$flag_source_system_tab_settings<-ifelse(master_df$source_system_tab=="settings",1,0)


#Creating source type based primary variables

master_df$flag_source_type_song<-ifelse(master_df$source_type=="song",1,0)
master_df$flag_source_type_song_based_playlist<-ifelse(master_df$source_type=="song-based-playlist",1,0)
master_df$flag_source_type_top_hits_for_artist<-ifelse(master_df$source_type=="top-hits-for-artist",1,0)
master_df$flag_source_type_topic_article_playlist<-ifelse(master_df$source_type=="topic-article-playlist",1,0)
master_df$flag_source_type_my_daily_playlist<-ifelse(master_df$source_type=="my-daily-playlist",1,0)
master_df$flag_source_type_online_playlist<-ifelse(master_df$source_type=="online-playlist",1,0)
master_df$flag_source_type_listen_with<-ifelse(master_df$source_type=="listen-with",1,0)
master_df$flag_source_type_local_library<-ifelse(master_df$source_type=="local-library",1,0)
master_df$flag_source_type_local_playlist<-ifelse(master_df$source_type=="local-playlist",1,0)
master_df$flag_source_type_album<-ifelse(master_df$source_type=="album",1,0)
master_df$flag_source_type_artist<-ifelse(master_df$source_type=="artist",1,0)


#For artist
master_df$flag_source_type_system_tab_artist_my_library<-master_df$flag_source_type_artist*master_df$flag_source_system_tab_my_library

#For listen with
master_df$flag_source_type_system_tab_listen_with<-master_df$flag_source_system_tab_listen_with*master_df$flag_source_type_listen_with


#For local library
master_df$flag_source_type_system_tab_local_library_my_library<-master_df$flag_source_type_local_library*master_df$flag_source_system_tab_my_library
master_df$flag_source_type_system_tab_local_library_discover<-master_df$flag_source_type_local_library*master_df$flag_source_system_tab_discover

#For local playlist
master_df$flag_source_type_system_tab_local_playlist_my_library<-master_df$flag_source_type_local_playlist*master_df$flag_source_system_tab_my_library

#For online playlist
master_df$flag_source_type_system_tab_online_playlist_discover<-master_df$flag_source_type_online_playlist*master_df$flag_source_system_tab_discover

#For song
master_df$flag_source_type_system_tab_song_search<-master_df$flag_source_type_song*master_df$flag_source_system_tab_search



#For song based playlist
master_df$flag_source_type_system_tab_song_based_playlist_discover<-master_df$flag_source_type_song_based_playlist*master_df$flag_source_system_tab_discover


train_data<-master_df[master_df$type=='train',]
test_data<-master_df[master_df$type=='test',]


#training and test data creation
train_data_xgb1<-master_df[master_df$type=="train",c(8:ncol(master_df))]
test_data_xgb1<-master_df[master_df$type=="test",c(8:ncol(master_df))]


train_data$target <- factor(train_data$target, levels = c(0,1), ordered = TRUE)
ydata <- as.numeric(train_data$target)-1

xdata <- Matrix(as.matrix(train_data_xgb1), sparse = TRUE)

xdata_test_final <- Matrix(as.matrix(test_data_xgb1), sparse = TRUE)

#####################Analysis done########################


#####################XGBoost start########################
# xgboost parameters
param <- list("objective" = "multi:softprob",    # multiclass classification 
              "num_class" = 2,    # number of classes 
              "eval_metric" = "merror",    # evaluation metric 
              "nthread" = 8,   # number of threads to be used 
              "max_depth" = 16,    # maximum depth of tree 
              "eta" = 0.3,    # step size shrinkage 
              "gamma" = 0,    # minimum loss reduction 
              "subsample" = 1,    # part of data instances to grow tree 
              "colsample_bytree" = 1,  # subsample ratio of columns when constructing each tree 
              "min_child_weight" = 12  # minimum sum of instance weight needed in a child 
              )


bst.cv <- xgb.cv(param=param, data=xdata, label=ydata, 
              nfold=2, nrounds=30, prediction=TRUE, verbose=TRUE)






min.merror.idx = which.min(bst.cv$evaluation_log[,test_merror_mean]) 


xgb <- xgboost(param=param, data=xdata, label=ydata,
                           nrounds=min.merror.idx, verbose=TRUE)


pred_xgb <- predict(xgb, xdata_test_final, reshape = TRUE)


pred_xgb2 <- as.data.frame(pred_xgb)
names(pred_xgb2) <- c("zero","one")
pred_xgb2$id <- test_data$id

pred_xgb2$target <- ifelse(pred_xgb2$zero>pred_xgb2$one,0,1)

pred_xgb3<-pred_xgb2[,c(3,4)]

write.csv(pred_xgb3, "predictions.csv", row.names = FALSE)





