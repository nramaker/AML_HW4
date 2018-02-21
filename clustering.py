
def load_data(file):
    return []

def produce_clusters(data):
    return []

def plot_dendogram(data):
    return []

#main entry
if __name__ == "__main__":
    print(" ##### AML HW4 Clusterererer  ##### ")
    print("")
    print("### Loading Data ...")
    data = load_data('data.csv')
    print("### Producing Clusters")
    clusters = produce_clusters(data)
    print("### Plotting Dendrogram")
    plot_dendogram(clusters)