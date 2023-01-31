import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from sklearn.datasets import make_moons

X, Y = make_moons(n_samples=20, noise=0)
print(X)
print(Y)

G = nx.Graph()
for i,node in enumerate(X):
    G.add_node(i, pos=(node[0],node[1]))

for i,node in enumerate(X):
    for j in range(i+1,len(X)):
        dx = node[0] - X[j][0]
        dy = node[0] - X[j][1]
        dist = (dx**2 + dy**2)**0.5
        G.add_edge(i,j,weight=dist)    

#L = nx.laplacian_matrix(G)

def matPrintView(matrix):
    size = matrix.shape
    for i in range(size[0]):
        for j in range(size[1]):
            print(matrix[i,j], " ",end = '')
        print()

#matrix.A returns matrix view
#print(L.A)
#eigW, eigV = np.linalg.eig(L.A)
#print("Largest eigenvalue:", max(e))
#print("Smallest eigenvalue:", min(e))

#print(eigW)
#print(eigV)

new_x = []
new_y = []

for r in X:
    new_x.append(r[0])
    new_y.append(r[1])

plt.scatter(new_x, new_y, c=Y)
plt.savefig("moons.png")

pos=nx.get_node_attributes(G,'pos')
ax = plt.subplot(121)
nx.draw(G, pos)
plt.savefig('graph.png')
