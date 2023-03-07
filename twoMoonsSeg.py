import helper as util
from sklearn.datasets import make_moons
import numpy as np
import networkx as nx
import time

def run_test(n, dt, c, ε, iterations):

    #Placeholder for Graph Time, segmentation Time and segmentation accuracy
    results = [] 

    ## ----------- DATASET --------------##
    X, Y = make_moons(n_samples=n, noise=0.1)
    ## ----------------------------------##


    ## ------CONSTRUCTING-GRAPH----------##
    start1 = time.time()

    moonsGraph = util.makeMoonsGraph(X)
    moonsGraph = util.n_nearest_Neighbours(moonsGraph,10)

    ## ----------------------------------##


    ## ------LAPLACIAN-AND-EIGENS--------##
    Lnorm = nx.normalized_laplacian_matrix(moonsGraph)
    eigvalues, eigVectors = np.linalg.eig(Lnorm.A)
    eigens = util.sortEigens(eigvalues, eigVectors)
    end1 = time.time()
    ## ----------------------------------##

    ## ----------SEGMENTATION------------##
    nodes = list(moonsGraph.nodes)

    start2 = time.time()

    seg1 = util.ginzburg_landau_segmentation(nodes, eigens["values"], eigens["vectors"], dt, c, ε, iterations)

    end2 = time.time()

    start3 = time.time()

    seg2 = util.second_eigenvector_segmentation(eigens["vectors"][1])

    end3 = time.time()

    total1 = 0
    total2 = 0
    for i in range(0,len(seg1)):
        if seg1[i] == Y[i]:
            total1 += 1
        if seg2[i] == Y[i]:
            total2 += 1


    accuracy1 = (total1/len(seg1) ) * 100
    if accuracy1 < 50:
        accuracy1 = 100 - accuracy1
    
    accuracy2 = (total2/len(seg2) ) * 100
    if accuracy2 < 50:
        accuracy2 = 100 - accuracy2
    ## ----------------------------------##

    ##--------------RESULTS--------------##
    
    results.append(end1 - start1)
    results.append(end2 - start2)
    results.append(end3 - start3)
    results.append(accuracy1)
    results.append(accuracy2)
    
    return results

def test_harness():
    results = {
        "graphTime": [],
        "seg1Time": [],
        "seg2Time": [],
        "seg1Accuracy": [],
        "seg2Accuracy": []
    }
    for i in range(0,10):
        res = run_test(1000, 0.1, 1, 2, 200)
        results["graphTime"].append(res[0])
        results["seg1Time"].append(res[1])
        results["seg2Time"].append(res[2])
        results["seg1Accuracy"].append(res[3])
        results["seg2Accuracy"].append(res[4])
        print(i, "-- COMPLETE")
    
    print("Average Graph Time                  -- ", np.mean(results["graphTime"]))
    print("Average Segmentation Time (GL)      -- ", np.mean(results["seg1Time"]))
    print("Average Segmentation Accuracy (GL)  -- ", np.mean(results["seg1Accuracy"]))
    print("Average Segmentation Time (φ_2)     -- ", np.mean(results["seg2Time"]))
    print("Average Segmentation Accuracy (φ_2) -- ", np.mean(results["seg2Accuracy"]))