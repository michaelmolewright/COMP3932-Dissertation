
import networkx as nx
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import matplotlib

def makeMoonsGraph(X):
    
    G = nx.Graph()
    for i,node in enumerate(X):
        G.add_node(i, pos=(node[0],node[1]))

    for i,node in enumerate(X):
        for j in range(i+1,len(X)):
            dx = node[0] - X[j][0]
            dy = node[1] - X[j][1]
            dist = (dx**2 + dy**2)**0.5
            w = 2.781 ** ( ((dist ** 2)) / 2)
            G.add_edge(i,j,weight = w)
    return G

def findBestEigen(eigVecs, Correct_labels, X):
    new_x = []
    new_y = []

    for r in X:
        new_x.append(r[0])
        new_y.append(r[1])
    kmeans = KMeans(n_clusters=2)

    for count, eigVec in enumerate(eigVecs):
        new_data = eigVec.reshape(-1,1)

        kmeans.fit(new_data)
        correct = 0
        for i, label in enumerate(Correct_labels):
            if label == kmeans.labels_[i]:
                correct += 1
            
        accuracy = correct/len(Correct_labels)
        print("EigenVector ", count, " Accuracy = ", accuracy)
        plt.scatter(new_x, new_y, c=kmeans.labels_)
        plt.savefig("moons.png")