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

def task3():
    labels = list_labels()
    print("labels {}".format(labels))
    print("")
    print("###Loading Data set")
    dirs = list_dirs("./HMP_Dataset", labels)
    #TODO stop doing this load
    #TODO train kmeans from other loaded data
    # all_data = []
    # for directory in dirs:
    #     data = load_and_join_data(directory)
    #     all_data.append(list(data))
    # #split data into test and training
    # train_X, test_X, train_Y, test_Y = test_train_split(all_data, labels, .20)

    # print("train_X[0].shape {}".format(np.asarray(train_X[0]).shape))
    # print("test_X[0].shape {}".format(np.asarray(test_X[0]).shape))
    # print("train_Y[0].shape {}".format(np.asarray(train_Y[0]).shape))
    # print("test_Y[0].shape {}".format(np.asarray(test_Y[0]).shape))
    
    # kmeans = train_kmeans_classifier(train_X, k=k, chunk_size=chunk_size )


    all_chunks = []
    all_labels = []
    for i in range(0, len(dirs)):
        cls_chunks, cls_labels = load_chunk_and_label(dirs[i], labels[i], chunk_size=chunk_size)
        all_chunks = np.append(all_chunks, np.asarray(cls_chunks), axis=0)
        all_labels = np.append(all_labels, np.asarray(cls_chunks), axis=0)
        # all_chunks.append(bt_chunks)
        # all_labels.append(bt_labels)
        # print("")
        # print("label: {}".format(labels[i]))
        # print("chunks shape: {}".format(np.asarray(bt_chunks).shape))
        # print("labels shape: {}".format(np.asarray(bt_labels).shape))
    
    print("Combined chunks and labels.")
    print("chunks shape: {}".format(np.asarray(all_chunks).shape))
    print("labels shape: {}".format(np.asarray(all_labels).shape))

    train_X, test_X, train_Y, test_Y = test_train_split(all_chunks, all_labels, .20)
    
    print("train_X.shape {}".format(np.asarray(train_X).shape))
    print("test_X.shape {}".format(np.asarray(test_X).shape))
    print("train_Y.shape {}".format(np.asarray(train_Y).shape))
    print("test_Y.shape {}".format(np.asarray(test_Y).shape))

    print("")
    print("train_X[0] {}".format(train_X[0]))
    kmeans = train_kmeans_classifier(train_X, k=k, chunk_size=chunk_size )

    
    
    #try to predict train_x[0]
    # print("Building Histograms for class {}".format(train_Y[0][0]))
    # histograms = predict_cluster_histogram(train_X[0], kmeans, chunk_size=chunk_size, k=k)
    # print("Finished")
    #classify new signal
    #       cut signal into pieces
    #       find closest cluster center from dictionary
    #       build histogram of cluster centers
    
    pass

def build_all_hisitograms(chunks, labels, ):
    pass

def load_chunk_and_label(directory, label, chunk_size=32):
    all_chunks = []
    all_labels = []
    files = list_files(directory)
    for f in files:
        # print("Loading and chunking data from {}".format(f))
        data = genfromtxt(f, delimiter=' ')
        data = chunkify(data, chunk_size=chunk_size)
        all_chunks.append(list(data))
        all_labels.append(label)
    return all_chunks, all_labels

def predict_cluster_histogram(chunks, clusterer, k=14):
    histogram = np.empty(k, int)
    predictions = clusterer.predict(np.asarray(chunks))
    print("Predictions {} {}".format(np.asarray(predictions).shape, predictions))
    # print("P2 {} {}".format(p2.shape, p2))
    for pred in predictions:
        histogram[pred] = histogram[pred]+1
    print(histogram)
    return histogram

def train_kmeans_classifier(data, k=3, chunk_size=32):
    print("")
    print("Training KMeans Clusterer k={}, chunk_size={}".format(k, chunk_size))
    # training_chunks = mass_chunkify(data, chunk_size=32)
    
    #       clustering with kmeans
    print("Training KMeans")
    kmeans = KMeans(n_clusters=k, random_state=seed).fit(data)
    return kmeans

# def mass_chunkify(data, chunk_size=32):
#     print("")
#     print("Chunking data into chunks of size {}".format(chunk_size))
#     all_chunks = np.empty((0,chunk_size), int)
#     for obs_class in data:
#         chunks = chunkify(obs_class, chunk_size=chunk_size)
#         all_chunks = np.append(all_chunks, np.asarray(chunks), axis=0)
#     return all_chunks

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
    X = []
    Y = []
    print("Labels length {}".format(len(labels)))
    #first join data from every category
    for i in range(0, 2):
    # for i in range(0, len(labels)):
        # print("Splitting {} data into test/train, test split = {}".format(labels[i], percent_test))
        print("")
        print("Combining X.shape {} with data[i].shape {}".format(np.asarray(X).shape, np.asarray(data[i]).shape))
        X.extend(data[i])
        print("New shape {}".format(np.asarray(X).shape))
        print("Combining Y.shape {} with labels[i].shape {}".format(np.asarray(Y).shape, np.asarray(labels[i]).shape))
        Y.extend(labels[i])
        print("New shape {}".format(np.asarray(Y).shape))

    print("After joining all data, X.shape {}".format(np.asarray(X).shape))
    print("After joining all data, Y.shape {}".format(np.asarray(Y).shape))
    # print("After joining all data, X[0].shape {}".format(np.asarray(X[0]).shape))
    # print("After joining all data, Y[0].shape {}".format(np.asarray(Y[0]).shape))

    # print("")
    # print(X[0])


    #split once 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=percent_test, random_state=seed)
    return X_train, X_test, y_train, y_test

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
    trimmed = list_files(parent_dir)
    data = []
    for t in trimmed:
        my_data = genfromtxt(t, delimiter=' ')
        for line in my_data:
            data.append(line)
    return data

def list_files(parent_dir):
    files = []
    #get all files in the directory, excluding _MODEL directories
    x = [os.path.join(r,file) for r,d,f in os.walk(parent_dir) for file in f]
    #filter out files in 
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
            files.append(path)
    return files

#main entry
if __name__ == "__main__":
    print(" ##### AML HW4 Clusterererer  ##### ")
    task3()

