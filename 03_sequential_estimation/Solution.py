# Author: Bjarki Laxdal - Bjarki18 AT ru.is 
# Date: 05.09.22
# Project: 03_sequential_estimation
# Acknowledgements: This project was done in collaboration with Gunnlaug Margret, Aron Ingi, Aisha Regina, Steinunn Bjorg, Fanney Einarsdottir, Ignas
#


from turtle import update
from tools import scatter_3d_data, bar_per_axis

import matplotlib.pyplot as plt
import numpy as np


def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    var: float
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    return np.random.multivariate_normal(mean,(var**2)*np.identity(k),n)

def update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    return np.array(mu+(x-mu)/n)

def _plot_sequence_estimate():
    data = gen_data(300,3,np.array([0, 1, -1]),np.sqrt(3))
    estimates = [np.array([0, 0, 0])]
    for i in range(data.shape[0]):
        estimates.append(update_sequence_mean(estimates[i],data[i],i+1))
    plt.plot([e[0] for e in estimates][1:], label='First dimension')
    plt.plot([e[1] for e in estimates][1:], label='Second dimension')
    plt.plot([e[2] for e in estimates][1:], label='Third dimension')
    plt.legend(loc='upper center')
    plt.show()


def _square_error(y, y_hat):
    err = np.power((y - y_hat),2)
    mean = []
    for i in range(err.shape[0]):
        mean.append(np.mean(err[i]))

    return mean

def _plot_mean_square_error():
    data = gen_data(300,3,np.array([0, 1, -1]),np.sqrt(3))
    estimates = [np.array([0, 0, 0])]
    for i in range(data.shape[0]):
        estimates.append(update_sequence_mean(estimates[i],data[i],i+1))
    

    results = []
    for j in range(len(estimates)):
        results.append(_square_error([0,1,-1],estimates[j]))

    plt.plot([np.mean(x) for x in results[1:]])
    plt.show()
    

# Naive solution to the independent question.
def gen_changing_data(
    n: int,
    k: int,
    start_mean: np.ndarray,
    end_mean: np.ndarray,
    var: float
) -> np.ndarray:
    # remove this if you don't go for the independent section
    mean_v = np.linspace(start_mean,end_mean,n)
    
    ret_arr = np.ndarray((n,k))
    for i in range(n):
        ret_arr[i,:] = gen_data(1,k,mean_v[i],var)
    
    return ret_arr

def _plot_changing_sequence_estimate():
    # remove this if you don't go for the independent section
    n = 500
    k = 3
    start_mean = [0, 1, -1]
    end_mean = [1, -1, 0]
    data = gen_changing_data(n,k,start_mean,end_mean,1)
    mean_v = np.linspace(start_mean,end_mean, n)

    estimates = [np.array([0, 0, 0])]
    for i in range(data.shape[0]):
        estimates.append(update_sequence_mean(estimates[i],data[i],i+1))
    
    estimates = estimates[1:]
    results = []
    for j in range(len(estimates)):
        results.append(_square_error(mean_v[j],estimates[j]))

    results = results[1:]
    plt.figure(1)
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')
    plt.plot([e[2] for e in estimates], label='Third dimension')
    plt.legend(loc='upper center')
    plt.show()

    plt.figure(2)
    plt.plot([r[0] for r in results], label='First dimension')
    plt.plot([r[1] for r in results], label='Second dimension')
    plt.plot([r[2] for r in results], label='Third dimension')
    plt.legend(loc='upper center')
    plt.show()


# if __name__ == "__main__":
    # np.random.seed(1234)
    # _plot_changing_sequence_estimate()
    