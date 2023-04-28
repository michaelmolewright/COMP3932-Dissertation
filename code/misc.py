import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def g(x,sig):
    e = 2.71828
    return np.power( e, -(x**2) / (2*sig**2 ) )

def plot_gausian_func():
    '''
    Function to plot the guassian function in matplotlib
    '''
    matplotlib.use('Agg')
    x = np.linspace(-20, 20, 100)

    plt.plot(x, g(x,2), color='red',label='σ = 2' )
    plt.plot(x, g(x,5), color='blue',label='σ = 5')
    plt.plot(x, g(x,10), color='green',label='σ = 10')
    plt.legend()

    plt.savefig("../plots/GaussianFunction")

