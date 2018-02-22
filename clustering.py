
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from scipy.spatial import distance
from scipy.spatial.distance import pdist
from sklearn.metrics import euclidean_distances

def load_data(file):
    print("Loading samples from file: {}".format(file))
    data = pd.read_csv(file)
    
    # print("Full data set is {} lines.".format(len(data)))

    countries = data['Country'].as_matrix()
    vectors = data.T[1:].T.as_matrix()
    return vectors, countries

def cluster_and_plot(vectors, labels, description, link_type='average',):
    #X = [[i] for i in vectors]
    X = euclidean_distances(vectors)
    Z = linkage(X, method=link_type)
    fig = plt.figure(figsize=(12, 8))
    dn = dendrogram(Z, labels=labels)
    plt.title(description)
    plt.show()
    
def task1():
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

def task2():
    pass

def task3():
    pass

#main entry
if __name__ == "__main__":
    print(" ##### AML HW4 Clusterererer  ##### ")
    task1()
    task2()
    task3()