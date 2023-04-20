import helper as util
from sklearn.datasets import make_moons
from scipy.sparse.linalg import eigsh
import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
import matplotlib

import buildGraph as bd
import segmentation as sgm

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

def run_test_with_plot(n, dt, c, ε, iterations, path):
    #Placeholder for Graph Time, segmentation Time and segmentation accuracy
    results = [] 
    print("Started")
    ## ----------- DATASET --------------##
    X, Y = make_moons(n_samples=n, noise=0.1)
    ## ----------------------------------##


    ## ------CONSTRUCTING-GRAPH----------##
    start1 = time.time()

    moonsGraph = util.makeMoonsGraph(X)
    moonsGraph = util.n_nearest_Neighbours(moonsGraph,10)
    print(moonsGraph.number_of_edges())

    ## ----------------------------------##


    ## ------LAPLACIAN-AND-EIGENS--------##
    Lnorm = nx.normalized_laplacian_matrix(moonsGraph)
    eigvalues, eigVectors = np.linalg.eig(Lnorm.A)
    eigens = util.sortEigens(eigvalues, eigVectors)
    end1 = time.time()

    #E,V = eigsh(Lnorm, 20, which = 'SM')
    #E = E[:,np.newaxis]
    print("Graphs + Eigens Complete")

    Eval = np.asarray(eigens["values"][:20])
    Evec = np.asarray(eigens["vectors"][:20])
    Evec = Evec.T
    Eval = Eval[:,np.newaxis]
    print(Evec.shape)
    print(Eval.shape)
    ## ----------------------------------##

    ## ----------SEGMENTATION------------##
    nodes = list(moonsGraph.nodes)

    start2 = time.time()

    seg1 = util.ginzburg_landau_segmentation_two(nodes, Eval, Evec, dt, c, ε, iterations)
    #seg1 = util.gl_zero_means_eig(V,E, .1, V[:,1])
    #seg1 = util.ginzburg_landau_segmentation(nodes, eigens["values"],     eigens["vectors"], dt, c, ε, iterations)
    print("GL segmentation complete")

    end2 = time.time()

    start3 = time.time()

    seg2 = util.second_eigenvector_segmentation(eigens["vectors"][1])
    print("second Eigen vector segmentation complete")

    end3 = time.time()
    ## ----------------------------------##

    ## -------------RESULTS--------------##
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

    ##---------------PLOT----------------##
    matplotlib.use('Agg')
    new_x = []
    new_y = []

    for r in X:
        new_x.append(r[0])
        new_y.append(r[1])

    print(len(X), len(new_x) )
    plt.scatter(new_x, new_y, c=seg1)
    plt.savefig("./plots/GL_Seg.png")

    plt.scatter(new_x, new_y, c=seg2)
    plt.savefig("./plots/Eigen2_Seg.png")

    ##-----------------------------------##

    print("\nGraph Construction Time     -- ", end1 - start1)
    print("Segmentation Time (GL)      -- ", end2 - start2)
    print("Segmentation Accuracy (GL)  -- ", accuracy1)
    print("Segmentation Time (φ_2)     -- ", end3 - start3)
    print("Segmentation Accuracy (φ_2) -- ", accuracy2)
    print("PLEASE SEE PLOT FOLDER FOR PLOT RESULTS")
    
def playground(n, k):
    X, Y = make_moons(n_samples=n, noise=0.1)

    moonsGraph = util.makeMoonsGraph(X)
    moonsGraph = util.n_nearest_Neighbours(moonsGraph,10)

    Lnorm = nx.normalized_laplacian_matrix(moonsGraph)
    eigvalues, eigVectors = np.linalg.eig(Lnorm.A)
    eigens = util.sortEigens(eigvalues, eigVectors)
    print("Graphs + Eigens Complete")

    seg = util.shi_malek_segmentation(k, eigens["vectors"], X)

    matplotlib.use('Agg')
    new_x = []
    new_y = []

    for r in X:
        new_x.append(r[0])
        new_y.append(r[1])


    plt.scatter(new_x, new_y, c=seg, marker=".")
    #plt.savefig("./plots/GL_Seg.png")
    plt.savefig("./plots/SHI_Seg.png")

def plot_two_moons(X, seg, path):
    matplotlib.use('Agg')
    plt.cla()
    new_x = []
    new_y = []

    for r in X:
        new_x.append(r[0])
        new_y.append(r[1])
    
    new_c = []

    for val in seg:
        if val == 1:
            new_c.append("red")
        else:
            new_c.append("green")

    print(len(X), len(new_x) )
    plt.scatter(new_x, new_y, c=new_c, marker=".")
    plt.savefig(path)

def accuracy(trueVals, predictedVals):
    accuracies = []
    totalRight = 0
    for i in range(len(trueVals)):
        if trueVals[i] == predictedVals[i]:
            totalRight += 1
    acc = totalRight / len(trueVals)
    if acc < 0.5:
        acc = 1-acc
    return acc

    
def experiment(samples, noise, n, vals, plot=False):
    
    results = {
        "totalT" : [],
        "graphT" : [],
        "lapT" : [],
        "segT" : [],
        "acc_scores" : []
    }

    for i in range(n):
        X, Y = make_moons(n_samples=samples, noise=noise)
        graphBuilder = bd.graphBuilder()
        segmenter = sgm.segment()
        
        #---------------Graph-------------------#
        
        graphBuilder.setup(X)

        graphTimeStart = time.time()
        graphBuilder.local_scaling(10)
        graphTimeEnd = time.time()

        lapTimeStart = time.time()
        segmenter.setup(graphBuilder.graph)
        lapTimeEnd = time.time()

        segTimeStart = time.time()
        seg = segmenter.gl_method(0.1,2,2,500,20)
        segTimeEnd = time.time()

        acc = accuracy(Y, seg)   
        #---------------------------------------#    
        results["totalT"].append(segTimeEnd - graphTimeStart)
        results["graphT"].append(graphTimeEnd - graphTimeStart)
        results["lapT"].append(lapTimeEnd - lapTimeStart)
        results["segT"].append(segTimeEnd - segTimeStart)
        results["acc_scores"].append(acc)

    plot_two_moons(X, seg, '../plots/fieldler_two_moons.jpg')

    totalT = np.mean(results["totalT"])
    graphT = np.mean(results["graphT"])
    lapT = np.mean(results["lapT"])
    segT = np.mean(results["segT"])
    acc_scores = np.mean(results["acc_scores"])
    
    print("Noise        -- ", noise)
    #print("Value        -- ", val)
    print("totalT       -- ", totalT)
    print("graphT       -- ", graphT)
    #print("lapT         -- ", lapT )
    #print("segT         -- ", segT )
    print("acc_scores   -- ", acc_scores)
    print()

