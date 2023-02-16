
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



'''
Making functions for the convex splitting scheme
'''

#Initial function for 2 Moons graph
#returns 1 or -1 if value is above or below 0
def u_initial(second_eigen, x):
    
    mean = 0
    for i in second_eigen:
        mean += i
    mean /= len(second_eigen)

    val = second_eigen[x] - mean
    if val <= 0:
        return -1
    else:
        return 1

#returns the U(x) function for the nth iteration
def u_nth(a_k, phi_k):
    val = 0
    return val

#Initializing different starting variables

#initialize a
#a(0)k = int u(x) dot φk(x) dx.
def a_init():
    a = []
    return a

#initialize b
#e b(0)k = int [u0(x)]^^3 dot φk(x) dx.
def b_init():
    a = []
    return a

#initialize d
#d(0)k = 0
def d_init(λ):
    d = []
    for val in λ:
        d.append(val)
    return d

#initialize D
#Dk = 1 + dt ( λ_k + c)
def D_init(dt, λ, c):
    D = []
    for val in λ:
        D.append(1 + (dt * ( val + c)))
    return D



#Fidelity function
def fidelity(x):
    return 1