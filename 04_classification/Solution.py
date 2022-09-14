# Author: Bjarki Laxdal - Bjarki18 AT ru.is 
# Date: 05.09.22
# Project: 04_classification
# Acknowledgements: This project was done in collaboration with Gunnlaug Margret, Aron Ingi, Aisha Regina, Steinunn Bjorg, Fanney Einarsdottir, Ignas
#


from tkinter.filedialog import test
from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    '''
    ret_arr = []
    for i in range(features.shape[1]):
        ret_arr.append(np.mean(features[targets==selected_class][:,i]))
    return np.array(ret_arr)


def covar_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    '''    
    return np.cov(features[targets==selected_class],rowvar=False)


def likelihood_of_class(
    feature: np.ndarray,
    class_mean: np.ndarray,
    class_covar: np.ndarray
) -> float:
    '''
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    '''
    return multivariate_normal(mean=class_mean,cov=class_covar).pdf(feature)


def maximum_likelihood(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [], []
    for class_label in classes:
        means.append(mean_of_class(train_features,train_targets,class_label))
        covs.append(covar_of_class(train_features,train_targets,class_label))
    
    likelihoods = []
    for i in range(test_features.shape[0]):
        temp_arr = []
        for j in range(len(classes)):
            temp_arr.append(likelihood_of_class(test_features[i],means[j],covs[j]))
        likelihoods.append(temp_arr)
    return np.array(likelihoods)


def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    ret_arr = []
    for i in range(likelihoods.shape[0]):
        ret_arr.append(np.argmax(likelihoods[i]))

    return np.array(ret_arr)


def maximum_aposteriori(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum a posteriori for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [], []
    for class_label in classes:
        means.append(mean_of_class(train_features,train_targets,class_label))
        covs.append(covar_of_class(train_features,train_targets,class_label))
    
    Nk = np.bincount(train_targets)
    N = len(train_targets)

    likelihoods = []
    for i in range(test_features.shape[0]):
        temp_arr = []
        for j in range(len(classes)):
            temp_arr.append(likelihood_of_class(test_features[i],means[j],covs[j])*Nk[j]/N)
        likelihoods.append(temp_arr)
    return np.array(likelihoods)

def confusion_matrix(
    classes: np.ndarray,
    test_targets: np.ndarray,
    likelihood: list,
) -> np.ndarray:
    x = y = len(classes)
    matrix = np.zeros((x,y))
    guess = predict(likelihood)
    target = test_targets

    for n in range(len(target)):
        matrix[guess[n]][target[n]] += 1

    return matrix

def accuracy(
    likelihood: np.ndarray,
    test_targets: np.ndarray,
) -> float:
    guess = predict(likelihood)
    real = test_targets
    correct = 0
    for i in range(len(test_targets)):
        if guess[i] == real[i]:
            correct += 1

    return correct/len(test_targets)




# if __name__ == "__main__":
#     # s11
#     features, targets, classes = load_iris()
#     (train_features, train_targets), (test_features, test_targets)\
#     = split_train_test(features, targets, train_ratio=0.6)

#     # print(mean_of_class(train_features, train_targets, 0)) #-> [5.005 3.4425 1.4625 0.2575]

#     # print(covar_of_class(train_features, train_targets, 0))
#     # class_mean = mean_of_class(train_features, train_targets, 0)
#     # class_cov = covar_of_class(train_features, train_targets, 0)
#     # print(likelihood_of_class(test_features[0, :], class_mean, class_cov))
#     # print(maximum_likelihood(train_features, train_targets, test_features, classes))
#     likelihoods = maximum_likelihood(train_features, train_targets, test_features, classes)
#     # temp = predict(likelihoods)
#     apost=maximum_aposteriori(train_features, train_targets, test_features, classes)
#     # print(likelihoods)
#     print(confusion_matrix(classes,test_targets,likelihoods))
#     print(confusion_matrix(classes,test_targets,apost))
#     print(accuracy(likelihoods,test_targets))
#     print(accuracy(apost,test_targets))