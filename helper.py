
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
            w = 2.781 ** ( -((dist ** 2)) / 2)
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

#Sort the eigen Values and Vectors
def sortingFunction(list):
    return list[0]

def sortEigens(eigenValues, eigenVectors):
    sortedEigens = []
    eigensDict = {
        "values" : [],
        "vectors" : [],
        "key" : []
    }
    
    for i, val in enumerate(eigenValues):
        sortedEigens.append([val, eigenVectors[i], i])
    
    sortedEigens.sort(reverse=False , key=sortingFunction)

    #POTENTIALLY needs to turn first eigen vector into 1 vector and value to 0
    #sortedEigens[0][0] = 0
    #for j, val in enumerate(sortedEigens[0][1]):
    #    sortedEigens[0][1][j] = 1 

    for item in sortedEigens:
        eigensDict["values"].append(item[0])
        eigensDict["vectors"].append(item[1])
        eigensDict["key"].append(item[2])
    return eigensDict

#Fidelity function
def fidelity(x):
    return 1

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
def u_nth(a_k, φ_k, x):
    total = 0
    for i, a in enumerate(a_k):
        total += a * φ_k[i][x]
    return total

#Initializing different starting variables

#initialize a
#a(0)k = int u(x) dot φk(x) dx.
def a_init( nodes, eigenVectors):
    a = []
    total = 0
    for k in range(len(eigenVectors)):
        for x in range(len(nodes)):
            total += u_initial( eigenVectors[1], x) * eigenVectors[k][x]
        a.append(total)
    return a

# a_nth
def a_nth(a, b, d, D, dt, epsilon, c):
    new_a = []
    for k in range(len(a)):
        term1 = ( 1 + ( dt / epsilon) + (c * dt) ) * a[k] #a term
        term2 = ( dt / epsilon ) * b[k] # b term
        term3 = dt * d[k] # d term
        
        value = ( term1 - term2 - term3 ) / D[k]
        new_a.append(value)

    return new_a

#initialize b
#e b(0)k = int [u0(x)]^^3 dot φk(x) dx.
def b_init(nodes, eigenVectors):
    b = []
    total = 0
    for k in range(len(eigenVectors)):
        for x in range(len(nodes)):
            total += u_initial( eigenVectors[1], x) * eigenVectors[k][x]
        b.append(total)
    return b

#b_nth
#e b(0)k = int [u0(x)]^^3 dot φk(x) dx.
def b_nth(nodes, eigenVectors, a_k):
    b = []
    total = 0
    for k in range(len(eigenVectors)):
        for x in range(len(nodes)):
            total += (u_nth(a_k, eigenVectors, x)) * eigenVectors[k][x]
        b.append(total)
    return b

#initialize d
#d(0)k = 0
def d_init(λ):
    d = []
    for val in λ:
        d.append(0)
    return d

#d_nth

def d_nth(nodes, eigenVectors, a_k):
    d = []
    total = 0
    for k in range(len(eigenVectors)):
        for x in range(len(nodes)):
            total += (u_nth(a_k, eigenVectors, x) - u_initial(eigenVectors[1], x)) * eigenVectors[k][x]
        d.append(total)
    return d

#initialize D
#Dk = 1 + dt ( epsilon * λ_k + c)
def D_init(dt, λ, c, epsilon):
    D = []
    for val in λ:
        D.append(1 + (dt * ( (epsilon * val) + c)))
    return D


def segment(x):
    if x <= 0:
        return -1
    else:
        return 1