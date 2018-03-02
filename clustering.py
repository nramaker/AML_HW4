
import numpy as np
from numpy import genfromtxt
import pandas as pd
import os
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

seed = 42
k = 14
chunk_size = 32


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

def task3():
    labels = list_labels()
    print("labels {}".format(labels))
    print("")
    print("###Loading Data set")
    dirs = list_dirs("./HMP_Dataset", labels)
    all_data = []
    for directory in dirs:
        data = load_and_join_data(directory)
        all_data.append(list(data))
    #split data into test and training
    train_X, test_X, train_Y, test_Y = test_train_split(all_data, labels, .20)

    print("train_X[0].shape {}".format(np.asarray(train_X[0]).shape))
    print("test_X[0].shape {}".format(np.asarray(test_X[0]).shape))
    print("train_Y[0].shape {}".format(np.asarray(train_Y[0]).shape))
    print("test_Y[0].shape {}".format(np.asarray(test_Y[0]).shape))
    
    #TODO build dictionary
    #cut signals into fixed size
    # training_chunks = mass_chunkify(train_X, chunk_size=32)
    # print("Training_chunks.shape {}".format(np.array(training_chunks).shape))
    # print("Training_chunks[0] {}".format(np.array(training_chunks[0])))
    
    #       clustering with kmeans
    # print("Training KMeans Classifier")
    #kmeans = KMeans(n_clusters=k, random_state=seed).fit(training_chunks)
    kmeans = train_kmeans_classifier(train_X, k=k, chunk_size=chunk_size )
    #classify new signal
    #       cut signal into pieces
    #       find closest cluster center from dictionary
    #       build histogram of cluster centers
    
    pass

def train_kmeans_classifier(data, k=3, chunk_size=32):
    print("")
    print("Training KMeans Clusterer k={}, chunk_size={}".format(k, chunk_size))
    training_chunks = mass_chunkify(data, chunk_size=32)
    # print("Training_chunks.shape {}".format(np.array(training_chunks).shape))
    # print("Training_chunks[0] {}".format(np.array(training_chunks[0])))
    
    #       clustering with kmeans
    print("Training KMeans")
    kmeans = KMeans(n_clusters=k, random_state=seed).fit(training_chunks)
    return kmeans

def mass_chunkify(data, chunk_size=32):
    print("")
    print("Chunking data into chunks of size {}".format(chunk_size))
    all_chunks = np.empty((0,chunk_size), int)
    for obs_class in data:
        chunks = chunkify(obs_class, chunk_size=chunk_size)
        # print("About to concat : A.shape {} B.shape {}".format(np.asarray(all_chunks).shape, np.asarray(chunks).shape))
        all_chunks = np.append(all_chunks, np.asarray(chunks), axis=0)
    return all_chunks

def chunkify(data, chunk_size=32):
    chunks = []

    iter = 0
    chunk = []
    for reading in data:
        for point in reading:
            chunk.append(point)
            iter +=1
            if iter >= chunk_size:
                chunks.append(list(chunk))
                chunk = []
                iter = iter%chunk_size
    return chunks

def test_train_split(data, labels, percent_test=.20):
    print("")
    print("### Splitting Data")
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for i in range(0, len(labels)):
        print("Splitting {} data into test/train, test split = {}".format(labels[i], percent_test))
        #construct labels
        X=data[i]
        y = [[]] * len(X)
        y[0].append(labels[i])
        #split into training and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=percent_test, random_state=seed)
        train_data.append(list(X_train))
        test_data.append(list(X_test))
        train_labels.append(list(y_train))
        test_labels.append(list(y_test)) 

    return train_data, test_data, train_labels, test_labels

def list_labels():
    labels = ["Brush_teeth","Climb_stairs","Comb_hair","Descend_stairs","Drink_glass","Eat_meat","Eat_soup","Getup_bed","Liedown_bed",
    "Pour_water","Sitdown_chair","Standup_chair","Use_telephone","Walk"]
    return labels

def list_dirs(parent_dir, labels):
    dirs = []
    for label in labels:
        dirs.append(parent_dir+"/"+label)
    return dirs

def load_and_join_data(parent_dir):
    print("Loading data from {}".format(parent_dir))
    #get all files in the directory, excluding _MODEL directories
    x = [os.path.join(r,file) for r,d,f in os.walk(parent_dir) for file in f]
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
    data = []
    for t in trimmed:
        my_data = genfromtxt(t, delimiter=' ')
        for line in my_data:
            data.append(line)
    return data


#main entry
if __name__ == "__main__":
    print(" ##### AML HW4 Clusterererer  ##### ")
    # agg_clustering()
    # k_means()
    task3()

    # data = [ [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    # chunks = chunkify(data, chunk_size=5)
    # print(chunks)