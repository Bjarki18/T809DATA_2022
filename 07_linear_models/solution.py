# Author: Bjarki Laxdal 
# Date: 30.09.22
# Project: 07_linear_models
# Acknowledgements: This project was done in collaboration with Gunnlaug Margret, Aron Ingi, Aisha Regina, Steinunn Bjorg, Fanney Einarsdottir, Iris Fridriksdottir
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.

import numpy as np
import matplotlib.pyplot as plt

from tools import load_regression_iris
from scipy.stats import multivariate_normal


def mvn_basis(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float
) -> np.ndarray:
    '''
    Multivariate Normal Basis Function
    The function transforms (possibly many) data vectors <features>
    to an output basis function output <fi>
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional
    data vectors.
    * mu: [MxD] matrix of M D-dimensional mean vectors defining
    the multivariate normal distributions.
    * sigma: All normal distributions are isotropic with sigma*I covariance
    matrices (where I is the MxM identity matrix)
    Output:
    * fi - [NxM] is the basis function vectors containing a basis function
    output fi for each data vector x in features
    '''
    (N,D) = features.shape
    M = mu.shape[0]
    fi = np.zeros((N,M))
    for i in range(fi.shape[1]):
        fi[:, i] = multivariate_normal.pdf(features, mu[i], sigma)

    return fi


def _plot_mvn():
    X, t = load_regression_iris()
    N, D = X.shape
    M, sigma = 10, 10
    mu = np.zeros((M, D))
    for i in range(D):
        mmin = np.min(X[i, :])
        mmax = np.max(X[i, :])
        mu[:, i] = np.linspace(mmin, mmax, M)
    fi = mvn_basis(X, mu, sigma)

    for i in range(fi.shape[1]):
        plt.plot(fi[:,i])

    plt.xlabel("n of features")
    plt.ylabel("basis")
    plt.show()


def max_likelihood_linreg(
    fi: np.ndarray,
    targets: np.ndarray,
    lamda: float
) -> np.ndarray:
    '''
    Estimate the maximum likelihood values for the linear model

    Inputs :
    * Fi: [NxM] is the array of basis function vectors
    * t: [Nx1] is the target value for each input in Fi
    * lamda: The regularization constant

    Output: [Mx1], the maximum likelihood estimate of w for the linear model
    '''

    return np.linalg.inv(fi.T.dot(fi) + (lamda * np.identity(fi.shape[1]))).dot(fi.T.dot(targets))
    


def linear_model(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float,
    w: np.ndarray
) -> np.ndarray:
    '''
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional data vectors.
    * mu: [MxD] matrix of M D dimensional mean vectors defining the
    multivariate normal distributions.
    * sigma: All normal distributions are isotropic with s*I covariance
    matrices (where I is the MxM identity matrix).
    * w: [Mx1] the weights, e.g. the output from the max_likelihood_linreg
    function.

    Output: [Nx1] The prediction for each data vector in features
    '''

    return np.matmul(mvn_basis(features,mu,sigma),w)




# if __name__ == "__main__":
#     X, t = load_regression_iris()
#     N, D = X.shape
#     M, sigma = 10, 10
#     mu = np.zeros((M, D))
#     for i in range(D):
#         mmin = np.min(X[i, :])
#         mmax = np.max(X[i, :])
#         mu[:, i] = np.linspace(mmin, mmax, M)
#     fi = mvn_basis(X, mu, sigma)
#     # print(fi)
#     _plot_mvn()
#     # print(fi)
#     lamda = 0.001
#     wml = max_likelihood_linreg(fi, t, lamda)
#     prediction = linear_model(X, mu, sigma, wml)

#     x = [x for x in range(len(t))]
#     plt.plot(x,prediction)
#     plt.plot(x,t)
#     plt.legend(["prediction","actual"])
#     plt.show()
