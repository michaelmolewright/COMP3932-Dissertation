import networkx as nx
import matplotlib.pyplot as plt
import matplotlib

G = nx.Graph()
G.add_node("A")
G.add_node("B")
G.add_node("C")
G.add_node("D")

G.add_edge("A","B", weight=3)
G.add_edge("A","C", weight=5)
G.add_edge("B","C", weight=1)

lp1 = nx.laplacian_matrix(G)
print(lp1.get_shape())

lp2 = nx.laplacian_matrix(G, weight="weight")
print(lp2)


#ax = plt.subplot(121)
#nx.draw(G, with_labels=True)
#plt.savefig('graph.svg')
