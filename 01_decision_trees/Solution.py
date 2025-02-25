# Author: Bjarki Laxdal Baldursson - Bjarki18 AT ru.is
# Date: 22.08.22
# Project: 01_decision_trees
# Acknowledgements: Done in collaboration with Aron Ingi, Steinunn Bjorg, Fanney Einarsd, Ignas and Iris
#


from dis import dis
from posixpath import split
from typing import Union
from xml.sax.handler import feature_external_ges
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix

from tools import load_iris, split_train_test

def prior(targets: np.ndarray, classes: list) -> np.ndarray:
    '''
    Calculate the prior probability of each class type
    given a list of all targets and all class types
    '''
    
    samples_count = len(targets)
    if samples_count == 0:
        samples_count = 1
        
    class_probs = []
    for c in classes:
        class_count = 0
        for t in targets:
            if t == c:
                class_count +=1
        class_probs.append( class_count / samples_count )

    return class_probs


def split_data(
    features: np.ndarray,
    targets: np.ndarray,
    split_feature_index: int,
    theta: float
) -> Union[tuple, tuple]:
    '''
    Split a dataset and targets into two seperate datasets
    where data with split_feature < theta goes to 1 otherwise 2
    '''
    features_1 = features[features[:,split_feature_index] < theta]
    targets_1 = targets[features[:,split_feature_index] < theta]

    features_2 = features[features[:,split_feature_index] >= theta]
    targets_2 = targets[features[:,split_feature_index] >= theta]

    return (features_1, targets_1), (features_2, targets_2)

def gini_impurity(targets: np.ndarray, classes: list) -> float:
    '''
    Calculate:
        i(S_k) = 1/2 * (1 - sum_i P{C_i}**2)
    '''

    sum_val = 0
    for item in prior(targets,classes):
        sum_val += item**2

    return 1/2 * (1 - (sum_val))

def weighted_impurity(
    t1: np.ndarray,
    t2: np.ndarray,
    classes: list
) -> float:
    '''
    Given targets of two branches, return the weighted
    sum of gini branch impurities
    '''
    g1 = gini_impurity(t1,classes)
    g2 = gini_impurity(t2,classes)
    n = t1.shape[0] + t2.shape[0]
    return (t1.shape[0]*g1/n)+(t2.shape[0]*g2/n)

def total_gini_impurity(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    split_feature_index: int,
    theta: float
) -> float:
    '''
    Calculate the gini impurity for a split on split_feature_index
    for a given dataset of features and targets.
    '''
    (_, t1), (_, t2) = split_data(features, targets, split_feature_index, theta)
    weight = weighted_impurity(t1,t2,classes)
    return weight

def brute_best_split(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    num_tries: int
) -> Union[float, int, float]:
    '''
    Find the best split for the given data. Test splitting
    on each feature dimension num_tries times.

    Return the lowest gini impurity, the feature dimension and
    the threshold
    '''
    best_gini, best_dim, best_theta = float("inf"), None, None
    # iterate feature dimensions
    for i in range(features.shape[1]):
        # create the thresholds
        thetas = np.linspace(features[:,i].min(),features[:,i].max(),num_tries+2)[1:-1]
        # iterate thresholds
        for theta in thetas:
            temp_gini = total_gini_impurity(features,targets,classes,i,theta)
            if temp_gini < best_gini:
                best_gini = temp_gini
                best_dim = i
                best_theta = theta

    return best_gini, best_dim, best_theta

class IrisTreeTrainer:
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        classes: list = [0, 1, 2],
        train_ratio: float = 0.8
    ):
        '''
        train_ratio: The ratio of the Iris dataset that will
        be dedicated to training.
        '''
        (self.train_features, self.train_targets),\
            (self.test_features, self.test_targets) =\
            split_train_test(features, targets, train_ratio)

        self.classes = classes
        self.tree = DecisionTreeClassifier()

    def train(self):
        self.tree.fit(self.train_features,self.train_targets)

    def accuracy(self):
        guess = self.guess()
        real = self.test_targets
        correct_guess = 0
        for i in range(len(real)):
            if guess[i] == real[i]:
                correct_guess += 1
        return correct_guess/len(real)
        

    def plot(self):
        plot_tree(self.tree)
        plt.show()

    def plot_progress(self):
        # Independent section
        # Remove this method if you don't go for independent section.
        accuracy = []
        for i in range(1,len(self.train_features)):
            self.tree.fit(self.train_features[:i][:],self.train_targets[:i])
            accuracy.append(self.accuracy())

        plt.plot(range(1,len(self.train_features)),accuracy)
        plt.show()

    def guess(self):
        return self.tree.predict(self.test_features)

    def confusion_matrix(self):
        guess = self.guess()
        target = self.test_targets
        return confusion_matrix(target,guess)
