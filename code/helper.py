import networkx as nx
from sklearn.cluster import KMeans
import random
import numpy as np

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import matplotlib

def makeMoonsGraph(X):
    
    G = nx.Graph()
    for i,node in enumerate(X):
        G.add_node(i, pos=(node[0],node[1]))

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

def dist(a,b):
    return (a**2 + b**2)**0.5

def n_nearest_Neighbours(G, M):
    node = 0

    all_distances = []
    for i in range(0,len(list(G.nodes))):
        init_pos = G.nodes[node]['pos']
        distances = []
        for n in list(G.nodes):
            distances.append(dist(init_pos[0] - G.nodes[n]['pos'][0] , init_pos[1] - G.nodes[n]['pos'][1]))
        distances.sort()
        #val = quickselect(distances, M) #May not be much better
        all_distances.append([node, distances[M]] )
        node += 1
    #all_distances.sort(key=sortingFunction)
    for i in range(0,len(list(G.nodes))):
        for n in range(i+1,len(list(G.nodes))):
            if n != i:
                d = dist(G.nodes[i]['pos'][0] - G.nodes[n]['pos'][0] , G.nodes[i]['pos'][1] - G.nodes[n]['pos'][1])
                G.add_edge(n, i, weight= 2.781 ** ( -((d ** 2)) / (all_distances[n][1] * all_distances[i][1] )))
    return G

def gaussian_weights(G, τ):
    for i in range(0,len(list(G.nodes))):
        for j in range(i,len(list(G.nodes))):
            if j != i:
                d = dist(G.nodes[i]['pos'][0] - G.nodes[j]['pos'][0] , G.nodes[i]['pos'][1] - G.nodes[j]['pos'][1])
                G.add_edge(i, j, weight= 2.781 ** ( -((d ** 2)) / τ ))
    return G

'''
Making functions for the convex splitting scheme
'''

#Sort the eigen Values and Vectors
def sortingFunction(list):
    return list[0]

def sortingFunction2(list):
    return list[0][0]

def sortEigens(eigenValues, eigenVectors):
    sortedEigens = []
    eigensDict = {
        "values" : [],
        "vectors" : [],
        "key" : []
    }
    
    for i, val in enumerate(eigenValues):
        sortedEigens.append([val, eigenVectors[:,i], i])
    
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
def u_initial(second_eigen, nodes):
    
    mean = 0
    for i in second_eigen:
        mean += i
    mean /= len(second_eigen)

    results = []
    for node in nodes:
        val = second_eigen[node] - mean
        if val <= 0:
            results.append(-1)
        else:
            results.append(1)
    return np.asarray(results)

#returns the U(x) function for the nth iteration
def u_nth(a_k, φ_k, x):
    total = 0
    for i, a in enumerate(a_k):
        total += a * φ_k[i][x]
    return total

def u_nth_cubed(a_k, φ_k, x):
    total = 0
    for i, a in enumerate(a_k):
        total += (a * φ_k[i][x])**3
    return total

#Initializing different starting variables

#initialize a
#a(0)k = int u(x) dot φk(x) dx.
def a_init( nodes, eigenVectors, first_u):
    a = []
    total = 0
    for k in range(0,20):
        for x in range(len(nodes)):
            total += first_u[x] * eigenVectors[k][x]
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
def b_init(nodes, eigenVectors, first_u):
    b = []
    total = 0
    for k in range(0,20):
        for x in range(len(nodes)):
            total += first_u[x] * eigenVectors[k][x]
        b.append(total)
    return b

#b_nth
#e b(0)k = int [u0(x)]^^3 dot φk(x) dx.
def b_nth(nodes, eigenVectors, results):
    b = []

    for k in range(0,20):
        total = 0
        for x in range(len(nodes)):
            total += results[x] * eigenVectors[k][x]
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

def d_nth(nodes, eigenVectors, results, first_u):
    d = []
    for k in range(0,20):
        total = 0
        for x in range(len(nodes)):
            total += (results[x] - first_u[x]) * eigenVectors[k][x]
        d.append(total)
    return d

#initialize D
#Dk = 1 + dt ( epsilon * λ_k + c)
def D_init(dt, λ, c, epsilon):
    D = []
    for val in λ:
        D.append(1 + ( dt * ( (epsilon * val) + c ) ) )
    return D


def segment(x):
    if x <= 0:
        return 0
    else:
        return 1

def second_eigenvector_segmentation(φ_2):
    kmeans = KMeans(n_clusters=2, n_init=10)

    kmeans.fit(φ_2.reshape(-1, 1))
    
    return kmeans.labels_

def get_all_u_nth(a_k, eigenVectors, nodes):
    results = []
    total = 0
    results_cubed = []
    for node in nodes:
        x = u_nth(a_k, eigenVectors, node)
        results.append( x )
        #results_cubed.append(x**3)
        total += x

    #Mean Constraint      int u(x)dx = 0,
    '''
    if total < -3:
        val = (total) / len(nodes)
        for r in range(0, len(results)):
            results[r] = results[r] - val
            results_cubed.append(results[r]**3)
    elif total > 3:
        val = (total) / len(nodes)
        for r in range(0, len(results)):
            results[r] = results[r] - val
            results_cubed.append(results[r]**3)
    else:
        for r in range(0, len(results)):
            results_cubed.append(results[r]**3)
    '''

    return [results, results_cubed]

def ginzburg_landau_segmentation(nodes, eigenValues, eigenVectors, dt, c, epsilon, iterations):
    segmentation = []
    first_u = u_initial(eigenVectors[1], nodes)
    a_k = a_init(nodes, eigenVectors, first_u)
    b_k = b_init(nodes, eigenVectors, first_u)
    d_k = d_init(eigenVectors)
    D_k = D_init(dt, eigenValues, c, epsilon)
    for iteration in range(0,iterations):

        a_k = a_nth( a_k, b_k, d_k, D_k, dt, epsilon, c )

        results = get_all_u_nth( a_k, eigenVectors, nodes )
        if iteration < 1000:
            b_k = b_nth( nodes, eigenVectors, results[0] )
        else:
            b_k = b_nth( nodes, eigenVectors, results[1] )

        d_k = d_nth( nodes, eigenVectors, results[0], first_u )
        
        print(iteration)

    
    for node in nodes:
        segmentation.append(segment(u_nth(a_k, eigenVectors, node)))

    return segmentation

def ginzburg_landau_segmentation_test(nodes, eigenValues, eigenVectors, dt, c, epsilon, iterations):
    segmentation = []
    u_init = np.asarray(eigenVectors[1])
    
    phi = np.asarray(eigenVectors[:20])
    a_k = np.dot(u_init, phi.T)
    b_k = np.dot(np.power(u_init,3), phi.T)
    D_k = np.asarray(eigenValues[:20])
    D_k = 1 + dt*(epsilon*D_k + c)
    d_k = np.zeros(len(D_k))

    temp1 = (1+ dt/epsilon + c*dt)
    u = u_init.copy()
    u_diff = 1
    i = 0
    while (i<iterations) and (u_diff > 1e-5):
        u_old = u.copy()
        a_k = np.divide( (temp1*a_k) -  ((dt * b_k)/epsilon) - (dt * d_k), D_k )

        u = np.dot(a_k, phi) # mean constraint

        b_k = np.dot(np.power(u,3), phi.T)

        d_k = np.dot( (u - u_init) , phi.T)

        u = u - np.mean(u)

        u_diff = (abs(u-u_old)).sum()
        print(u[1])
        i += 1
    
    labels = u
    labels[labels<0] = 0
    labels[labels>0] = 1

    u_init = labels

    return labels

def _diffusion_step_eig(v,V,E,dt):

    if len(v.shape) > 1:
        return np.dot(V,np.divide(np.dot(V.T,v),(1+dt*E)))
    else:
        u_new = np.dot(V,np.divide(np.dot(V.T,v[:,np.newaxis]),(1+dt*E)))
        return u_new.ravel()

def _gl_forward_step(u_old,dt,eps):
    v = u_old-dt/eps*(np.power(u_old,3)-u_old) #double well explicit step
    return v


def ginzburg_landau_segmentation_two(nodes, eigenValues, eigenVectors, dt, c, epsilon, iterations):
    segmentation = []
    #u_init = u_initial(eigenVectors[:, 1], nodes)
    u_init = eigenVectors[:, 1]

    i = 0
    u_new = u_init.copy()
    u_diff = 1
    tol = 0.0001

    while (i<iterations) and (u_diff > tol):
        u_old = u_init.copy()
        w = u_old.copy()
        for k in range(10): # diffuse and threshold for a while
            v = _diffusion_step_eig(w,eigenVectors,eigenValues,epsilon*dt)
            w = v-np.mean(v) # force the 0 mean
        #v = _gl_forward_step(u_old,dt, epsilon)
        #v = v-np.mean(v)
        u_new = _gl_forward_step(w,dt, epsilon)
        u_diff = (abs(u_new-u_old)).sum()

        i = i+1
    
    for j in u_new:
        segmentation.append(segment(j))
    labels = u_new
    labels[labels<0] = -1
    labels[labels>0] = 1

    print(i, "  ", u_diff)
    return labels

def shi_malek_segmentation(k, eigenvectors):

    X = []
    for i in range(1, k+1):
        for j, value in enumerate(eigenvectors[i]):
            if i == 1:
                X.append([value])
            else:
                X[j].append(value)

    
    for i, row in enumerate(X):
        total = sum_squared(row)
        for j, val in enumerate(row):
            if total == 0:
                total = 1
            X[i][j] = val / (total**0.5)
    

    kmeans = KMeans(n_clusters=2)

    kmeans.fit(X)

    return kmeans.labels_

def sum_squared(arr):
    total = 0
    for i in arr:
        total += i**2
    return total

def plot_two_moons(n, dev, path):
    matplotlib.use('Agg')

    X, Y = make_moons(n_samples=n, noise=dev)
    new_x = []
    new_y = []
    col = []

    for r in X:
        new_x.append(r[0])
        new_y.append(r[1])
        col.append([0.02,0.02,0.02])

    plt.scatter(new_x, new_y, c=col, marker=".")
    plt.savefig(path)


def perona_freeman_segmentation(k, eigenvectors):

    X = []
    for i in range(1, k+1):
        for j, value in enumerate(eigenvectors[i]):
            if i == 1:
                X.append([value])
            else:
                X[j].append(value)


    kmeans = KMeans(n_clusters=2)

    kmeans.fit(X)

    return kmeans.labels_

def plot_two_moons( data, seg, path ):
    matplotlib.use('Agg')
    new_x = []
    new_y = []

    for r in data:
        new_x.append(r[0])
        new_y.append(r[1])

    plt.scatter(new_x, new_y, c=seg)
    plt.savefig(path)

def plot_bar(variables, vals, path):
    matplotlib.use('Agg')

    plt.xlabel('Similarity Function')
    plt.ylabel('Time (s)')
    
    plt.bar(variables, vals, color="gray")
    plt.savefig("../plots/" + path)
