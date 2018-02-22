
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

def load_data(file):
    print("Loading samples from file: {}".format(file))
    data = pd.read_csv(file, header=None)
    
    print("Full data set is {} lines.".format(len(data)))

    return data.iloc[1:], data.iloc[0]

def cluster_and_plot(vectors, labels, description, link_type='average',):
    X = [[i] for i in vectors]
    Z = linkage(X, method=link_type, metric='euclidean')
    fig = plt.figure(figsize=(10, 10))
    dn = dendrogram(Z)
    fig.title = description
    plt.xlabel(countries)
    plt.show()

# def produce_clusters(data):
#     return data

# def plot_dendogram(data):
#     return data
    
#main entry
if __name__ == "__main__":
    print(" ##### AML HW4 Clusterererer  ##### ")
    print("")
    print("### Loading Data ...")
    data, labels = load_data('data.csv')
    countries = data[0]
    vectors = data.T[1:].T

    # print("vectors {}".format(vectors))
    # print("countries {}".format(countries))
 
    print("### Producing Clusters with Single Link")
    cluster_and_plot(vectors, countries, link_type='single', description="Single Link")

    print("### Producing Clusters with Average Link")
    cluster_and_plot(vectors, countries, description="Average Link")

    print("### Producing Clusters with Complete Link")
    cluster_and_plot(vectors, countries, link_type='complete', description="Complete Link")