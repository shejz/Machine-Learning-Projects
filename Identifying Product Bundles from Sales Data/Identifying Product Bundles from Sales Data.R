

df = read.csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00396/Sales_Transactions_Dataset_Weekly.csv')

data = read.csv('https://s3.amazonaws.com/dspython.dezyre.com/notebook_files/02-06-18-12-47-40/Sales_Transactions_Dataset_Weekly.csv',header = T,sep = ',')

library(cluster);library(cclust);library(fastcluster)
library(caret);library(mlbench)

####Data Pre-processing
par(mfrow=c(2,3))
apply(df[,c(-1)],2,boxplot)

#computing quantiles
quant<-function(x){quantile(x,probs=c(0.95,0.90,0.99))}
out1<-sapply(df[,c(-1)],quant)

# normalized data
r_df = df[,56:107]

#raw data set
r_df1 = df[,0:51]

#Checking outliers after removal
apply(r_df,2,boxplot)

#scaling dataset
#r_df_scaled<-as.matrix(scale(r_df[,c(-1,-2)]))
#head(r_df_scaled)


# there are different techniques for clustering
# 1. Ppartition based methods
# 2. Hirarchical Methods
# 3. Model based methods

# kmeans, k-median, k-mode
# agglomorative, divisive
# expectation maximization based methods

# kmeans algorithm is availble in 3 different libraries
# Rcmdr- KMeans
# stats- Kmeans
# AMAP- kmeans

# Steps in k-means algorithm
# 1. to identify the value of K
# 2. how to select k
# 3. decide the distance function
# 4. decide the iterations
# 5. get clusters
# 6. make profiling

# to select the value of K, a screeplot approach is ebing used
# screeplot shows the within group of sum of squares with the 
# corresponding number of clusters
# As the number of cluster increases the within group sum of squares(wss) will decrease
# eventually a point will come where the wss will not drop further
# that is the optimal scenario when the cluster is giving us best result

#Selecting Optimum Number of clusters, from the screeplot by looking at the 
# elbow point, once it is decided then create clusters and report the cluster goodness

# how do we know the clusters that we have created are good clusters
# to know the goodness of the clusters, we need to run a function called
# silhoutte score, this score ranges from -1 to +1
# -1 being the worst clusters and +1 being the best clusters
# 0- being clusters with overlapping members

# once you get a cluster solution apply silhoutte score on the cluster result
# make conclusions around how good the cluster solution is

# knowing the distance function that works best is a difficult task, hence it is 
# recommended to try different other distance functions to calculate the 
# cluster membership, and compute the silhoutte score, the best will be 
# automatically decided

# within group sum of squares are the distances of individual data values
# from the cluster center points/ cenetroid and take the distacne, square the values
# sum the squared distances

library(Rcmdr)

sumsq<-NULL
#Method 1
par(mfrow=c(1,2))
for (i in 1:15) sumsq[i] <- sum(KMeans(r_df,
                                       centers=i,
                                       iter.max=500, 
                                       num.seeds=50)$withinss)

plot(1:15,sumsq,type="b", xlab="Number of Clusters", 
     ylab="Within groups sum of squares",main="Screeplot using Rcmdr")

#Method 2
for (i in 1:15) sumsq[i] <- sum(kmeans(r_df,
                                       centers=i,
                                       iter.max=500, 
                                       algorithm = "Forgy")$withinss)

plot(1:15,sumsq,type="b", xlab="Number of Clusters", 
     ylab="Within groups sum of squares",main="Screeplot using Stats")

#Kmeans Clustering
library(cluster);library(cclust)
set.seed(121)
km<-kmeans(r_df,
           centers=4,
           nstart=17,
           iter.max=500, 
           algorithm = "Forgy",
           trace = T)


#checking results
summary(km)
km$centers
km$withinss

#attaching cluster information
Cluster<-cbind(r_df,Membership=km$cluster)
aggregate(Cluster[,-53],list(Cluster[,53]),mean)

#plotting cluster info
clusplot(Cluster, 
         km$cluster, 
         cex=0.9,
         color=TRUE, 
         shade=TRUE,
         labels=4, 
         lines=0)

#Predicting new data for KMeans
predict.kmeans <- function(km, r_df)
{k <- nrow(km$centers)
n <- nrow(r_df)
d <- as.matrix(dist(rbind(km$centers, r_df)))[-(1:k),1:k]
out <- apply(d, 1, which.min)
return(out)}

#predicting cluster membership
Cluster$Predicted<-predict.kmeans(km,r_df)
table(Cluster$Membership,Cluster$Predicted)

#writing the result to a file  
write.csv(Cluster,"predout1.csv")

# cluster model deployment

# deplymnet using an API (restful API)
# using PMML scipt
# calling custom functions from external applications such as T-SQL

#pmml code
library(pmml);
library(XML);
pmml(km)

#Hierarchical Clustering-agglomorative method
dev.off()
hfit<-hclust(dist(r_df,method = "euclidean"),method="ward.D2")
par(mfrow=c(1,2))
plot(hfit,hang=-0.005,cex=0.7)

hfit<-hclust(dist(r_df,method = "manhattan"),method="mcquitty")
plot(hfit,hang=-0.005,cex=0.7)

hfit<-hclust(dist(r_df,method = "minkowski"),method="ward.D2")
plot(hfit,hang=-0.005,cex=0.7)

hfit<-hclust(dist(r_df,method = "canberra"),method="ward.D2")
plot(hfit,hang=-0.005,cex=0.7)

#method	
#the agglomeration method to be used. This should be (an unambiguous 
#abbreviation of) one of "ward.D", "ward.D2", "single", "complete", "average" 
#(= UPGMA), "mcquitty" (= WPGMA), "median" (= WPGMC) or "centroid" (= UPGMC).


#attaching cluster information
summary(hfit)

#Hierarchical Clustering-divisive method
dfit<-diana(r_df,
            diss=F,
            metric = "euclidean",
            stand=T,
            keep.data = F)
summary(dfit)
plot(dfit)

#compare the clustering models
si3 <- silhouette(cutree(hfit, k = 4), # k = 4 gave the same as pam() above
                  daisy(r_df))
plot(si3)

si3 <- silhouette(cutree(hfit, k = 4), # k = 4 gave the same as pam() above
                  diana(r_df,
                        diss=F,
                        metric = "euclidean",
                        stand=T,
                        keep.data = F))
plot(si3)

#cutting the tree into groups
g_hfit<-cutree(hfit,k=4)
table(g_hfit)
plot(hfit)
rect.hclust(hfit,k=4,border = "blue")

# next step is to find out the cluster goodness using silhoutte score on
# the raw dataset, instead of the normalized dataset

pr4 <- pam(r_df, 4)
str(si <- silhouette(pr4))
plot(si)
plot(si, col = c("red", "green", "blue", "purple"))# with cluster-wise coloring

ar <- agnes(r_df)
si3 <- silhouette(cutree(ar, k = 2), # k = 4 gave the same as pam() above
                  daisy(r_df))
plot(si3, nmax = 80, cex.names = 0.5)

# Model based clustering and how it works.

######model based clustering##################
library(mclust)
clus <- Mclust(r_df)
summary(clus)

# Plotting the BIC values:
plot(clus, data=r_df, what="BIC")

# The clustering vector:
clus_vec <- clus$classification
clus_vec

clust <- lapply(1:3, function(nc) row.names(r_df)[clus_vec==nc])  
clust   # printing the clusters

# This gives the probabilities of belonging to each cluster 
#for every object:

round(clus$z,2)
summary(clus, parameters = T)

#self organizing maps
library(kohonen)
som_grid <- somgrid(xdim = 20, ydim=20, topo="hexagonal")

som_model <- som(as.matrix(r_df))

plot(som_model, type="changes",col="blue")
plot(som_model, type="count")

plot(som_model, type="dist.neighbours")
plot(som_model, type="codes")

##############################################
install.packages("factoextra")
install.packages("cluster")
install.packages("magrittr")

library("cluster")
library("factoextra")
library("magrittr")

res.dist <- get_dist(r_df, stand = TRUE, method = "pearson")
fviz_dist(res.dist, 
          gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))

library("factoextra")
fviz_nbclust(r_df, kmeans, method = "gap_stat")
fviz_nbclust(r_df, kmeans, method = "silhouette")
fviz_nbclust(r_df, kmeans, method = "wss")

set.seed(123)
km.res <- kmeans(r_df, 6, nstart = 25)
# Visualize
library("factoextra")
fviz_cluster(km.res, data = r_df,
             ellipse.type = "convex",
             palette = "jco",
             ggtheme = theme_minimal())

# Compute PAM
library("cluster")
pam.res <- pam(r_df, 6)
# Visualize
fviz_cluster(pam.res)

#####Hierarchical clustering######
# Compute hierarchical clustering
res.hc <- r_df %>%
  scale() %>%                    # Scale the data
  dist(method = "euclidean") %>% # Compute dissimilarity matrix
  hclust(method = "ward.D2")     # Compute hierachical clustering
# Visualize using factoextra
# Cut in 4 groups and color by groups
fviz_dend(res.hc, k = 6, # Cut in four groups
          cex = 0.5, # label size
          k_colors = c("#2E9FDF", "#00AFBB", "#E7B800", "#FC4E07"),
          color_labels_by_k = TRUE, # color labels by groups
          rect = TRUE # Add rectangle around groups
)

#Assessing clustering tendency
#Hopkins statistic: If the value of Hopkins statistic is close to 1 
#(far above 0.5), then we can conclude that the dataset is significantly 
#clusterable.

gradient.color <- list(low = "steelblue",  high = "white")
r_df %>%    #
  scale() %>%     # Scale variables
  get_clust_tendency(n = 50, gradient = gradient.color)

#Determining the optimal number of clusters
set.seed(123)
# Compute
library("NbClust")
res.nbclust <- r_df %>%
  scale() %>%
  NbClust(distance = "euclidean",
          min.nc = 2, max.nc = 10, 
          method = "complete", index ="all")

# Visualize
library(factoextra)
fviz_nbclust(res.nbclust, ggtheme = theme_minimal())

#Clustering validation statistics
#The silhouette plot is one of the many measures for inspecting and validating 
#clustering results. Recall that the silhouette (Si) measures how similar an 
#object i is to the the other objects in its own cluster versus those in the 
#neighbor cluster. Si values range from 1 to - 1:

#A value of Si close to 1 indicates that the object is well clustered. In the 
#other words, the object i is similar to the other objects in its group.
#A value of Si close to -1 indicates that the object is poorly clustered, and 
#that assignment to some other cluster would probably improve the overall 
#results.

set.seed(123)
# Enhanced hierarchical clustering, cut in 3 groups
res.hc <- r_df %>%
  scale() %>%
  eclust("hclust", k = 6, graph = FALSE)
# Visualize with factoextra
fviz_dend(res.hc, palette = "jco",
          rect = TRUE, show_labels = FALSE)

fviz_silhouette(res.hc)

#Which samples have negative silhouette? To what cluster are they closer?
# Silhouette width of observations
sil <- res.hc$silinfo$widths
# Objects with negative silhouette
neg_sil_index <- which(sil[, 'sil_width'] < 0)
sil[neg_sil_index, , drop = FALSE]

#Advanced Clustering
#Hierarchical K-Means Clustering
df <- scale(r_df)
# Compute hierarchical k-means clustering
library(factoextra)
res.hk <-hkmeans(df, 6)
# Elements returned by hkmeans()
names(res.hk)

# Print the results
res.hk


# Visualize the tree
fviz_dend(res.hk, cex = 0.6, palette = "jco", 
          rect = TRUE, rect_border = "jco", rect_fill = TRUE)

# Visualize the hkmeans final clusters
fviz_cluster(res.hk, palette = "jco", repel = TRUE,
             ggtheme = theme_classic())

######Methods####
#1. K-means
#2. PAM
#3. HClucst
#4. Daisy
#5. Agnes
#6. Model based
#7. HKmeans
#8. Silhouette Score
#9. Viz of clusters
#10. Model nbclust
####################
#Net Step: Please re-run all the scripts for the raw columns
#raw dataset
r_df1 = df[,2:53]
names(r_df1)
#normalized dataset
r_df = df[,56:107]
names(r_df)

#please repeat the same script for both datasets