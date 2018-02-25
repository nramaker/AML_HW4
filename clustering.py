
import numpy as np
from numpy import genfromtxt
import pandas as pd
import os
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import euclidean_distances
from sklearn.cluster import KMeans

from matplotlib import pyplot as plt

def load_data(file):
    print("Loading samples from file: {}".format(file))
    data = pd.read_csv(file)
    
    # print("Full data set is {} lines.".format(len(data)))

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
        # df = pd.DataFrame()
        # df['countries'] = countries
        # df['labels'] = kmeans.labels_
        # print(df)
    
    plt.figure(1)
    
    plt.plot(ks,distances, marker='o')
        
    plt.title('Average Distance from Centroids for k Clusters')
    plt.legend()
    plt.grid(True)
    plt.show()

def task3():
    #TODO load data
    files = load_and_join_data('./HMP_Dataset')
    #TODO split data into test and training
    #TODO build dictionary
    #       cut signals into fixed size
    #       clustering with kmeans
    #classify new signal
    #       cut signal into pieces
    #       find closest cluster center from dictionary
    #       build histogram of cluster centers
    
    pass

def load_and_join_data(parent_dir):
    #get all files in the directory, excluding _MODEL directories
    x = [os.path.join(r,file) for r,d,f in os.walk(parent_dir) for file in f]
    # print(x)
    #filter out files in 
    trimmed = []
    for path in x:
        if path.find('_MODEL') >=0 :
            continue
        elif path.find('displayModel.m') >=0:
            continue
        elif path.find('displayTrial.m') >=0:
            continue
        elif path.find('MANUAL.txt') >=0:
            continue
        elif path.find('README.txt') >=0:
            continue
        else:
            trimmed.append(path)
    # print(trimmed)
    data = []
    # print("File {}".format(trimmed[0]))
    # my_data = genfromtxt(trimmed[0], delimiter=' ')
    #df = pd.read_csv(trimmed[0], header=None)
    # data.append(my_data)
    print(data)
    for t in trimmed:
        my_data = genfromtxt(t, delimiter=' ')
        for line in my_data:
            data.append(line)
        # print("my_data.shape {}".format(np.asarray(my_data).shape))
        #data.append(my_data)

    print("data shape {}".format(np.asarray(data).shape))
    print("data[0] {}".format(data[0]))
    return data


#main entry
if __name__ == "__main__":
    print(" ##### AML HW4 Clusterererer  ##### ")
    # agg_clustering()
    # k_means()
    task3()