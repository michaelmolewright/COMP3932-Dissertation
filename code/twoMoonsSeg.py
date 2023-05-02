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
import random


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
    plt.scatter(new_x, new_y, c=new_c, marker=".")
    plt.savefig(path)

def accuracy(trueVals, predictedVals):
    '''
    Calculates the accuracy of the segementation
    '''
    accuracies = []
    totalRight = 0
    for i in range(len(trueVals)):
        if trueVals[i] == predictedVals[i]:
            totalRight += 1
    acc = totalRight / len(trueVals)
    if acc < 0.5:
        acc = 1-acc
    return acc
    
def experiment(samples, noise, n, plot=False):
    '''
    Function that tests the 3 segmentation algorithms on the Two Moons data set
    '''
    
    results = {
        "totalT" : [],
        "graphT" : [],
        "lapT" : [],
        "segT" : [],
        "acc_scores1" : [],
        "acc_scores2" : [],
        "acc_scores3" : [],
        "iters":[],
        "seg2T" : [],
        "seg3T" : [],
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


        segTimeStart1 = time.time()
        seg1 = segmenter.ginzburg_landau_segmentation_method(0.1,1, 2, 500)
        segTimeEnd1 = time.time()
        results["segT"].append(segTimeEnd1 - segTimeStart1)

        segTimeStart2 = time.time()
        seg2 = segmenter.fielder_method()
        segTimeEnd2 = time.time()
        results["seg2T"].append(segTimeEnd2 - segTimeStart2)

        segTimeStart3 = time.time()
        seg3 = segmenter.perona_freeman_method(4)
        segTimeEnd3 = time.time()
        results["seg3T"].append(segTimeEnd3 - segTimeStart3)

        acc1 = accuracy(Y, seg1) 
        acc2 = accuracy(Y, seg2) 
        acc3 = accuracy(Y, seg3)

        plot_two_moons(X, seg1, '../plots/GL_two_moons.jpg')
        plot_two_moons(X, seg2, '../plots/fielder_method.jpg')
        plot_two_moons(X, seg3, '../plots/perona_freeman_method.jpg')   
        #---------------------------------------#    
        results["totalT"].append(segTimeEnd3 - graphTimeStart)
        results["graphT"].append(graphTimeEnd - graphTimeStart)
        results["lapT"].append(lapTimeEnd - lapTimeStart)
        results["segT"].append(segTimeEnd1 - segTimeStart1)
        results["seg2T"].append(segTimeEnd2 - segTimeStart2)
        results["seg3T"].append(segTimeEnd3 - segTimeStart3)
        results["acc_scores1"].append(acc1)
        results["acc_scores2"].append(acc2)
        results["acc_scores3"].append(acc3)
        #results["iters"].append(seg[1])
        print(i)

        
        #plot_two_moons(X, seg, '../plots/GL_two_moons.jpg')

    totalT = np.mean(results["totalT"])
    graphT = np.mean(results["graphT"])
    lapT = np.mean(results["lapT"])
    acc_scores1 = np.mean(results["acc_scores1"])
    acc_scores2 = np.mean(results["acc_scores2"])
    acc_scores3 = np.mean(results["acc_scores3"])
    #variance = np.var(results["acc_scores"])
    segT = np.mean(results["segT"])
    iters = np.mean(results["iters"])
    seg2T = np.mean(results["seg2T"])
    seg3T = np.mean(results["seg3T"])

    print("Fielder Method:")
    print("Time      -- ", seg2T)
    print("Accuracy  -- ", acc_scores2)
    print()
    print("Perona Freeman Method:")
    print("Time      -- ", seg3T)
    print("Accuracy  -- ", acc_scores3)
    print()
    print("GL Method:")
    print("Time      -- ", segT)
    print("Accuracy  -- ", acc_scores1)
    print()
    print()

