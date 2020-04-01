#pip install sklearn
#pip install dtw==1.3.3

import os
import pandas as pd
import collections

#define a Commodity object
Commodity = collections.namedtuple('Commodity', ['name', 'price', 'dates']) #look to add debt in at some point!

#get all csv file names in the Houly folder
commodity_csv_ist = os.listdir(os.path.join(os.getcwd(), 'commodity_CSV'))

masterCommodityList = []

for commodityCSV in commodity_csv_ist:

    if '.csv' not in commodityCSV:
        continue

    data = pd.read_csv(os.path.join(os.getcwd(), 'commodity_CSV', commodityCSV), low_memory=False)

    commodity = Commodity(commodityCSV.replace(".csv", ""),
            [float(price) for price in data.iloc[:,1]],
            data.iloc[:,[0]]) #make this into a datetime object!

    masterCommodityList.append(commodity)

import numpy as np
from dtw import dtw
import scipy.cluster.hierarchy as hac
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import fcluster
from sklearn import preprocessing
from scipy.cluster import hierarchy

'''
This puts the DTW into a function because we must provide the hierarchical clustering function with a defined function
'''

def DTW(X, Y):
  x = np.array(X)
  y = np.array(Y)

  euclidean_norm = lambda x, y: np.abs(x - y)

  #The important part here is w! It is the "window" within which the series can be warped.
  #Ex: a window of 4 will allow each series to match one point to any point within 4 measurements of the other

  # Set window to 12 to allow things to warp up to a year
  d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean_norm, w=12)

  #Shows the distance matrix
  '''
  import matplotlib.pyplot as plt

  plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
  plt.plot(path[0], path[1], 'w')
  plt.show()
  '''

  return d

# MUST MAKE SURE ALL PRICE LISTS ARE OF THE SAME LENGTH!!!
# This line also standardizes the data. use preprocessing.normalize(X) if you want to try it when normalized
clusteringList = []
indexNameList = []
for series in masterCommodityList:
    if len(series.price) == 302:
        clusteringList.append(preprocessing.scale([i for i in series.price]))
        indexNameList.append(series.name)
        clusteringList.append(preprocessing.scale([-1*i for i in series.price]))
        indexNameList.append(series.name + ' inverted')

X = np.array(clusteringList)

# Make the Clusters
#Z = hac.linkage(X, method='average', metric='euclidean')
#Z = hac.linkage(X, method = 'average', metric=DTW)

# Plot with Custom leaves
#hierarchy.dendrogram(Z, leaf_rotation=90, leaf_font_size=8, labels=indexNameList)
#plt.show()

metal = masterCommodityList[0]
oil = masterCommodityList[0]

for i in masterCommodityList:
    if "Copper" in i.name:  #Iron Ore, Tianjin (ODA)
        print('hello')
        metal = i
    if "Dubai" in i.name: #Brent Crude Oil
        oil = i
        print('hello')

print("The DTW distance map between " + metal.name + " and " + oil.name)

x = [i for i in metal.price]
y = [i for i in oil.price]

plt.plot(x)
plt.plot(y)
plt.show()

x = preprocessing.scale([i for i in metal.price])
y = preprocessing.scale([i for i in oil.price])

plt.plot(x)
plt.plot(y)
plt.show()

euclidean_norm = lambda x, y: np.abs(x - y)

d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean_norm, w=12)

# Visualise the accumulated cost and the shortest path
plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.show()
