                                # Kmeans Algorithm : 
   
Kmeans algorithm is an iterative algorithm that tries to partition the dataset into Kpre-defined distinct non-overlapping subgroups (clusters) where each data point belongs to only one group. It tries to make the intra-cluster data points as similar as possible while also keeping the clusters as different (far) as possible. It assigns data points to a cluster such that the sum of the squared distance between the data points and the cluster’s centroid (arithmetic mean of all the data points that belong to that cluster) is at the minimum. The less variation we have within clusters, the more homogeneous (similar) the data points are within the same cluster.

                # The way kmeans algorithm works is as follows :
                
1. Specify number of clusters K.
2. Initialize centroids by first shuffling the dataset and then randomly selecting K data points for the centroids without replacement.
3. Keep iterating until there is no change to the centroids. i.e assignment of data points to clusters isn’t changing.
4. Compute the sum of the squared distance between data points and all centroids.
5. Assign each data point to the closest cluster (centroid).
6. Compute the centroids for the clusters by taking the average of the all data points that belong to each cluster.


The approach kmeans follows to solve the problem is called Expectation-Maximization. The E-step is assigning the data points to the closest cluster. The M-step is computing the centroid of each cluster. Below is a break down of how we can solve it mathematically (feel free to skip it).


                                                # Implementation : 
We’ll use simple implementation of kmeans here to just illustrate some concepts. Then we will use sklearn implementation that is more efficient take care of many things for us.

                # Example :

import numpy as np
from numpy.linalg import norm


class Kmeans:
    '''Implementing Kmeans algorithm.'''

    def __init__(self, n_clusters, max_iter=100, random_state=123):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def initializ_centroids(self, X):
        np.random.RandomState(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids

    def compute_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)
        return centroids

    def compute_distance(self, X, centroids):
        distance = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = norm(X - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance

    def find_closest_cluster(self, distance):
        return np.argmin(distance, axis=1)

    def compute_sse(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))
    
    def fit(self, X):
        self.centroids = self.initializ_centroids(X)
        for i in range(self.max_iter):
            old_centroids = self.centroids
            distance = self.compute_distance(X, old_centroids)
            self.labels = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroids(X, self.labels)
            if np.all(old_centroids == self.centroids):
                break
        self.error = self.compute_sse(X, self.labels, self.centroids)
    
    def predict(self, X):
        distance = self.compute_distance(X, old_centroids)
        return self.find_closest_cluster(distance)


                                # Applications : 
                                
kmeans algorithm is very popular and used in a variety of applications such as market segmentation, document clustering, image segmentation and image compression, etc. The goal usually when we undergo a cluster analysis is either:

1.) Get a meaningful intuition of the structure of the data we’re dealing with.
2.) Cluster-then-predict where different models will be built for different subgroups if we believe there is a wide variation in the behaviors of different subgroups. An example of that is clustering patients into different subgroups and build a model for each subgroup to predict the probability of the risk of having heart attack.

                # In this post, we’ll apply clustering on two cases:
1.) Geyser eruptions segmentation (2D dataset).
2.) Image compression.

#################################################################################################################################

    #BOSTON HOUSING Data Classification Using KMeans Cluster Analysis Algorithim ( Distance based -partitional Clustering Algo )

import pandas as pd
from sklearn.datasets import load_boston
boston=load_boston()

ds=pd.DataFrame(boston.data,columns=boston.feature_names)
ds.head()

#1-hot encoding of RAD variable; because its categorical variable
#representing it as categorical variable
ds["RAD"]=ds["RAD"].astype("category")
#datatype of the ds
ds.dtypes

#now using df.get_dummies(); it will drop the original column also
#this method will automatically pick the categorical variable and apply 1-hot encoding
ds=pd.get_dummies(ds,prefix="RAD")
ds.head()

#now doing Scaling on AGE,TAX,B or on entire Dataset
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler();
scaler=scaler.fit(ds)

scaledData=scaler.transform(ds)

#now create the scaled dataframe from it
dss=pd.DataFrame(scaledData,columns=ds.columns)

#now perform the clusetring 
#step 1  cluster configuration to kind the k
#step 2 using the value of 'k', generate the cluster

#now to know the best value of 'k' 
# wss/bss vs k

#That is when k=2, wss=sum of all point with theri 2 centeroid individually 
#        i.e within clusterdistance ( this is inertia )
#    and   bwss means distance between centroid c1 and c2

#now when k=3, wss= sum of distance all point of culter and their centroid 
# the above wss is given by inertia of the cluster configuration
## but for bwss the sum of distance between 3 centroid.
## c1 to c2, c1 to c3 and c2 to c3

###when cluster configuration=4
##the bss= dist(c1,c2)+dist(c1,c3) +dist(c1,c4) + dist(c2,c3) +dist(c2,c4) +dist(c3,c4)

#so all possible combination we need to find out for all values of k


from sklearn.cluster import KMeans
from itertools import combinations_with_replacement

from itertools import combinations 
from scipy.spatial import distance
print(list(combinations_with_replacement("ABCD", 2)))

wss=[]
bss=[]
pairmap={}
dis=[]
d=0
distanceMap={}
for k in range(2,16):
    #perforiming  the cluster configuration
    clust=KMeans(n_clusters=k,random_state=0).fit(dss)
    wss.append(clust.inertia_)
    c=list(combinations(range(0,k), 2))
    print("Combinations ----------->",c)
    print("ClusterCenters Are Below----------->")
    dataFrameClusterCenter=pd.DataFrame(clust.cluster_centers_)
    print(pd.DataFrame(clust.cluster_centers_))
    print("The above are clusterCenters are for k==",k)
    pairmap[k]={"pairs":c}
    for i in c:
        #converting the tuple() to list using the list() method
        pair=list(i)
        print("pair is",pair)
        #extracting the index from the pair
        index1=pair[0]
        index2=pair[1]
        #print("row 1"); print(dataFrameClusterCenter.iloc[index1,:])
        #print("row 2"); print(dataFrameClusterCenter.iloc[index2,:])
        d=distance.euclidean(dataFrameClusterCenter.iloc[index1,:],
                             dataFrameClusterCenter.iloc[index2,:])
        print("distance",d)
        #appending the calculated distance between each pair of the cluster centers in a list
        dis.append(d)  
        distanceMap[k]={"distance":dis}
    #making the list empty for next k
    dis=[]
        
print("disstacne map for each k ")
print(distanceMap)   
print("wss for all k ")
print(wss)     


#sum the distance of between every cluster 
#summedDistance storing to bss list
bss=[]
import math
for i in range(2,16):
    value=distanceMap.get(i)
    print(value)
    list=value['distance']
    print(math.fsum(list))
    summedDistance=math.fsum(list)
    bss.append(summedDistance)
    
bss
#1. now we have bss for all the k 
bss
#2. now we have wss for all the k
wss
#but wss shal be sqrt(wss[i])
len(wss)
len(bss)
sqrtwss=[]
for i in range(0,len(wss)):
    sqrt=math.sqrt(wss[i])
    print(sqrt)
    sqrtwss.append(sqrt)

#so this sqrtwss shall be used
sqrtwss


#final ratio =sqrtwss/bss
ratio=[]
for i in range(0,len(sqrtwss)):
    #ratio.append(sqrtwss[i]/wss[i])
    ratio.append(sqrtwss[i]/bss[i])
    
    #So finally perforimg scatter plot of ratio vs k plot
#########################   ratio=(sqrtwss/bss) vs k plot ############################
ratio
del list
k=range(2,16)
k
k=list(k)
k
from matplotlib import pyplot as plt
plt.plot(k,ratio)
plt.xlabel("No of cluster k")
plt.ylabel("Ratio of sqrtwss/bss")
plt.show()


#plot of sqrtwss vs k
plt.plot(k,sqrtwss)
plt.xlabel("No of cluster k")
plt.ylabel("wss or sqrtwss")
plt.show()


#plot of bss vs k
plt.plot(k,bss)
plt.xlabel("No of cluster k")
plt.ylabel("bss")
plt.show()




############# Now as we knoe the optiomal value of k is 4, so 
############# So we now perform actual clustering of 506 observations and there scaled 
############ scaled and linear independence dataset

#our scaled dataset is represented by dss
dss.shape
#to find corelation matrix 
dss.corr()


#now performing the clustering
clust=KMeans(n_clusters=4,max_iter=500,random_state=0).fit(dss)

#now extract the clusterCenters
clusterCenter=clust.cluster_centers_

#convert clusterCenter to dataframe to do the cluster profilin
ccd=pd.DataFrame(clusterCenter,columns=dss.columns)

#ccd for cluster profilin
ccd
#so profiling details
#clusterId 1 is having the highest crime rate
# industry are more in clusterId 1              


#to see the labels i.e clusterId for each observation
labels=clust.labels_

#total labes;
len(labels)
clusterIds=list(labels)

#now perform the inverse Scaling
originalDataAsNumpy=scaler.inverse_transform(dss)
#converting numpy to dataset
originalDataset=pd.DataFrame(originalDataAsNumpy,columns=dss.columns)

#adding the labelled column to the originalDataset
originalDataset["Label"]=labels

#saving data on the system as OriginalData.csv
originalDataset.to_csv("yoursystem path\\originalData.csv")
#to see whether data contains the label or not
originalDataset.Label[0]

##### Now plotting the Classfication 
import pylab as pl
len=originalDataset.shape[0]
len
for i in range(0, len):
   if originalDataset.Label[i] == 0:
      c1 = pl.scatter(originalDataset.iloc[i,2],originalDataset.iloc[i,4],c='r', marker='+')
   elif originalDataset.Label[i]  == 1:
      c2 = pl.scatter(originalDataset.iloc[i,2],originalDataset.iloc[i,4],c='g',marker='o')
   elif originalDataset.Label[i]  == 2:
      c3 = pl.scatter(originalDataset.iloc[i,2],originalDataset.iloc[i,4],c='b',marker='*')
   elif originalDataset.Label[i] == 3:
      c4 = pl.scatter(originalDataset.iloc[i,2],originalDataset.iloc[i,4],c='y',marker='^')
pl.legend([c1, c2, c3,c4], ['c1','c2','c3','c4'])  
pl.title('Boston Data classification')
pl.show()
