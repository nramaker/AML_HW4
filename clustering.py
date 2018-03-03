
import numpy as np
from numpy import genfromtxt
import pandas as pd

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import euclidean_distances
from sklearn.cluster import KMeans

from matplotlib import pyplot as plt

def load_data(file):
    print("Loading samples from file: {}".format(file))
    data = pd.read_csv(file)

    countries = data['Country'].as_matrix()
    vectors = data.T[1:].T.as_matrix()
    return vectors, countries

def cluster_and_plot(vectors, labels, description, link_type='average',):
    X = euclidean_distances(vectors)
    Z = linkage(X, method=link_type)
    fig = plt.figure(figsize=(12, 8))
    dn = dendrogram(Z, labels=labels)
    plt.title(description)
    plt.show()
    
def agg_clustering():
    print("")
    print("### Task1")
    print("### Loading Data ...")
    vectors, countries = load_data('data.csv')
 
    print("### Producing Clusters with Single Link")
    cluster_and_plot(vectors, countries, link_type='single', description="Single Link")

    print("### Producing Clusters with Average Link")
    cluster_and_plot(vectors, countries, description="Average Link")

    print("### Producing Clusters with Complete Link")
    cluster_and_plot(vectors, countries, link_type='complete', description="Complete Link")

def k_means(k=3):
    print("### Task2")
    print("### Loading Data ...")
    vectors, countries = load_data('data.csv')

    ks = []
    distances = []
    for i in range(2, len(countries)):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(vectors)
        ks.append(i)
        distances.append(kmeans.inertia_/len(countries))
    
    plt.figure(1)
    
    plt.plot(ks,distances, marker='o')
        
    plt.title('Average Distance from Centroids for k Clusters')
    plt.legend()
    plt.grid(True)
    plt.show()


#main entry
if __name__ == "__main__":
    print(" ##### AML HW4 Clusterererer  ##### ")
    agg_clustering()
    k_means()
