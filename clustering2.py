import numpy as np
from numpy import genfromtxt
import pandas as pd
import os
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import itertools

from matplotlib import pyplot as plt

seed = 42
k = 10
chunk_size = 32
trees =30
depth =16 

def task3():
    labels = list_labels()
    dirs = list_dirs("./HMP_Dataset", labels)

    ks = []
    sizes = []
    training_accs = []
    test_accs = []
    for k_value in range(1,11):
        training_acc, test_acc = full_test(dirs, labels, chunk_size=chunk_size, k=k_value*3, tree_count=trees, tree_depth=depth)
        ks.append(k_value*3)
        sizes.append(chunk_size)
        training_accs.append(training_acc)
        test_accs.append(test_accs)
        print("For k={}, chunksize={}: {} {}".format(k_value*3, chunk_size, training_acc, test_acc))
    print("Finished")

def create_conf_matrix(predictions, truths, classes, title):
    cm = confusion_matrix(predictions, truths)
    np.set_printoptions(precision=2)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def full_test(dirs, labels, chunk_size=32, k=20, tree_count=30, tree_depth=16 ):
    # print("### Full Test chunk_size={}, k={}, tree_count={} tree_depth={}".format(chunk_size, k, tree_count,tree_depth))
    chunks = []
    chunk_labels = []
    for i in range(0, len(dirs)):
        cls_chunks, cls_labels = load_chunk_and_label(dirs[i], labels[i], chunk_size=chunk_size)
        chunks.append(cls_chunks)
        chunk_labels.append(cls_labels)

    train_X, test_X, train_Y, test_Y = test_train_split(chunks, chunk_labels, .20)
    
    combined_train_chunks = np.empty((0,chunk_size), int)
    combined_train_labels = []
    for i in range(0, len(train_X)):
        combined_train_chunks = np.append(combined_train_chunks, np.asarray(train_X[i]), axis=0)
        combined_train_labels.append(train_Y[i])

    kmeans = train_kmeans_classifier(combined_train_chunks, k=k, chunk_size=chunk_size )

    train_histograms = build_histograms(train_X, kmeans, k)

    classifier = fit_rand_forrest_classifier(train_histograms, np.asarray(train_Y), trees=trees, depth=depth)
    training_accuracy, train_pred = rf_predict_and_measure(classifier, train_histograms, np.asarray(train_Y), "Training Accuracy")
    # create_conf_matrix(train_pred, train_Y, labels, 'TRAINING matrix, k={}, chunk_size={}'.format(k, chunk_size))

    test_histograms = build_histograms(test_X, kmeans, k)
    testing_accuracy, test_preds = rf_predict_and_measure(classifier, test_histograms, test_Y, "Testing Accuracy")
    create_conf_matrix(test_preds,test_Y, labels, 'TESTING accuracy={}, k={}, chunk_size={}'.format(testing_accuracy[1],k, chunk_size))
    return training_accuracy, testing_accuracy

def fit_rand_forrest_classifier(features, labels, trees, depth):
    # print("### Fitting RandomForestClassifier with depth={}, trees={} on {} features and {} labels.".format(depth, trees, np.asarray(features).shape, np.asarray(labels).shape))
    rfc = RandomForestClassifier(max_depth=depth, n_estimators=trees)
    return rfc.fit(features,labels.ravel())

def rf_predict_and_measure(classifier, features, truths, description):
    predictions = classifier.predict(features)
    accurracy = calculate_accuracy(predictions, truths, description)
    tup = (description, accurracy)
    return tup, predictions

def calculate_accuracy(predictions, truth, name):
    total = len(predictions)
    right = np.count_nonzero(predictions==truth)
    return float(right)/total

def build_histograms(data, clusterer, k):
    histograms = []
    for i in range(0, len(data)):
        histogram = predict_cluster_histogram(data[i], clusterer, k=k)
        histograms.append(histogram)
    return histograms

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
    histogram = np.zeros(k, int)
    predictions = clusterer.predict(np.asarray(chunks))
    for pred in predictions:
        histogram[pred] = histogram[pred]+1
    return histogram

def train_kmeans_classifier(data, k=3, chunk_size=32):
    # print("### Training KMeans Clusterer k={}, chunk_size={}".format(k, chunk_size))
    kmeans = KMeans(n_clusters=k, random_state=seed).fit(data)
    return kmeans

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
    # print("### Splitting Data")
    X = []
    Y = []
    #first join data from every category
    for i in range(0, len(labels)):
        X.extend(data[i])
        Y.extend(labels[i])
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

