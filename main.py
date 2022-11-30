import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

G = nx.Graph()
G.add_node("A")
G.add_node("B")
G.add_node("C")

G.add_edge("A","B", weight=3)
G.add_edge("A","C", weight=5)
G.add_edge("B","C", weight=1)

lp1 = nx.laplacian_matrix(G)

def matPrintView(matrix):
    size = matrix.shape
    for i in range(size[0]):
        for j in range(size[1]):
            print(matrix[i,j], " ",end = '')
        print()

matPrintView(lp1)



#ax = plt.subplot(121)
#nx.draw(G, with_labels=True)
#plt.savefig('graph.svg')
