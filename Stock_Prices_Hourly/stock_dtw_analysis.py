#pip install sklearn
#pip install dtw==1.3.3

import os
import pandas as pd
import collections

#define a stock object
Stock = collections.namedtuple('Stock', ['symbol', 'price', 'volume', "dates"]) #look to add debt in at some point!

#get all csv file names in the Houly folder
hourlyStocks = os.listdir(os.path.join(os.getcwd(), 'Hourly'))

masterStockList = []

for stockCSV in hourlyStocks:

    data = pd.read_csv(os.path.join(os.getcwd(), 'Hourly', stockCSV), low_memory=False)

    stock = Stock(stockCSV.replace(".csv", ""),
            [float(closePrice) for closePrice in data['Close']],
            [float(Volume) for Volume in data['Volume']],
            data['Datetime']) #make this into a datetime object!

    masterStockList.append(stock)

import numpy as np
from dtw import dtw
import scipy.cluster.hierarchy as hac
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import fcluster
from sklearn import preprocessing

'''
This is a function for showing the calculated differences between two series. Copied and pasted from the internet.
'''

def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

'''
This puts the DTW into a function because we must provide the hierarchical clustering function with a defined function
'''

def DTW(X, Y):
  x = np.array(X)
  y = np.array(Y)

  euclidean_norm = lambda x, y: np.abs(x - y)

  #The important part here is w! It is the "window" within which the series can be warped.
  #Ex: a window of 4 will allow each series to match one point to any point within 4 measurements of the other

  # Set window to 45 since there are 9 measurements per day. This allows any point to warp a week in either direction
  d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean_norm, w=45)

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
X = np.array([preprocessing.scale(stock.price) for stock in masterStockList if len(stock.price) == 420])

# Make the Clusters
#Z = hac.linkage(X, method='average', metric='euclidean')
Z = hac.linkage(X[:100], method = 'average', metric=DTW)

# Dendrogram with more information (Key part is max_d)
plt.figure(figsize=(15, 10))
fancy_dendrogram(
  Z,
  leaf_rotation=90.,
  max_d=0.1,
  annotate_above=15,
)
plt.show()
