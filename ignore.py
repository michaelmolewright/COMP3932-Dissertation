''''''
n = 2000

start = time.time()
X, Y = make_moons(n_samples=n, noise=0.1)
end = time.time()
print(end - start," -- 1. Moons DataSet")

X = np.ndarray.tolist(X)
Y = np.ndarray.tolist(Y)

start = time.time()
'''newArr = []
for i,x in enumerate(X):
    newArr.append([x,Y[i]])
newArr.sort(key=util.sortingFunction2)

X = []
Y = []
for i in newArr:
    X.append(i[0])
    Y.append(i[1])
'''
end = time.time()
print(end - start," -- 2. Sorting X and Y")


moonsGraph = util.makeMoonsGraph(X)
start = time.time()
moonsGraph = util.n_nearest_Neighbours(moonsGraph,10)
end = time.time()
print(end - start," -- 3. Similarity Function")
#moonsGraph = util.gaussian_weights(moonsGraph, 2)
start = time.time()
Lnorm = nx.normalized_laplacian_matrix(moonsGraph)
end = time.time()
print(end - start," -- 4. Laplacian")
#matrix.A returns matrix view

#print(L.toarray())

#Get Eigen Values and vectors
eigvalues, eigVectors = np.linalg.eig(Lnorm.A)


'''
Eigens Dictionary structure 
Values = []
Vectors = []
Key = []
'''
start = time.time()
eigens = util.sortEigens(eigvalues, eigVectors)
end = time.time()
print(end - start," -- 5. Eigen Sorting")

c = 1
epsilon = 2
dt = 0.1
iterations = 100

'''            
for i in eigVectors:
    for j in eigVectors:
        print(round(np.dot(i, j), 2))
#initialize a

'''
new_x = []
new_y = []

for r in X:
    new_x.append(r[0])
    new_y.append(r[1])

nodes = list(moonsGraph.nodes)
start = time.time()
seg = util.ginzburg_landau_segmentation(nodes, eigens["values"], eigens["vectors"], 0.1, 1, 2, 200)
#seg = util.second_eigenvector_segmentation(eigens["vectors"][1])
end = time.time()
print(end - start," -- 6. Segmentation")
plt.scatter(new_x, new_y, c=seg)
plt.savefig("moons.png")

total = 0
for i in range(0,len(seg)):
    if seg[i] == Y[i]:
        total += 1

accuracy = (total/len(seg) ) * 100
if accuracy < 50:
    accuracy = 100 - accuracy
print("accuracy = ", accuracy, "%" )
'''
ax = plt.subplot(121)
nx.draw(moonsGraph, pos)
plt.savefig('graph.png')
'''