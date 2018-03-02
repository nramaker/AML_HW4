
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
    dirs = list_dirs("./HMP_Dataset", labels)
    all_data = []
    for directory in dirs:
        data = load_and_join_data(directory)
        all_data.append([data])
    #split data into test and training
    train_X, test_X, train_Y, test_Y = test_train_split(data, labels, .20)
    print("train_X.shape {}".format(np.asarray(train_X).shape))
    print("train[0] {}".format(train[0]))
    print("test_X.shape {}".format(np.asarray(test_X).shape))
    print("train_Y.shape {}".format(np.asarray(train_Y).shape))
    print("test_Y.shape {}".format(np.asarray(test_Y).shape))
    #TODO build dictionary
    #       cut signals into fixed size
    #       clustering with kmeans
    #classify new signal
    #       cut signal into pieces
    #       find closest cluster center from dictionary
    #       build histogram of cluster centers
    
    pass

def test_train_split(data, labels, percent_test=.20):
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
        train_data.append(X_train)
        test_data.append(X_test)
        train_labels.append(y_train)
        test_labels.append(y_test) 

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
    # print(data)
    for t in trimmed:
        my_data = genfromtxt(t, delimiter=' ')
        for line in my_data:
            data.append(line)

    print("data shape {}".format(np.asarray(data).shape))
    # print("data[0] {}".format(data[0]))
    return data


#main entry
if __name__ == "__main__":
    print(" ##### AML HW4 Clusterererer  ##### ")
    # agg_clustering()
    # k_means()
    task3()