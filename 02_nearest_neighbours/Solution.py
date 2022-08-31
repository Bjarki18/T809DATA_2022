# Author: Bjarki Laxdal Baldursson - Bjarki18 AT ru.is
# Date: 22.08.22
# Project: 02_nearest_neighbours
# Acknowledgements: Done in collaboration with Aron Ingi, Steinunn Bjorg, Fanney Einarsd, Ignas and Iris
#

from itertools import accumulate
from math import dist
from turtle import distance
import numpy as np
import matplotlib.pyplot as plt
import help
from tools import load_iris, split_train_test, plot_points

def euclidian_distance(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Calculate the euclidian distance between points x and y
    '''
    sum_val = 0
    for i in range(x.shape[0]):
        sum_val += (x[i]-y[i])**2

    return np.sqrt(sum_val)

d, t, classes = load_iris()
x, points = d[0,:], d[1:, :]
x_target, point_targets = t[0], t[1:]

def euclidian_distances(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    '''
    Calculate the euclidian distance between x and and many
    points
    '''
    distances = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        distances[i] = euclidian_distance(x,points[i])
    
    return distances

def k_nearest(x: np.ndarray, points: np.ndarray, k: int):
    '''
    Given a feature vector, find the indexes that correspond
    to the k-nearest feature vectors in points
    '''
    value: np.ndarray = np.argsort(euclidian_distances(x,points))
    return value[0:k]

def vote(targets, classes):
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    most_popular_class = 0
    count = 0
    targets = list(targets)

    for item in classes:
        target_count = targets.count(item)
        if count < target_count:
            count = target_count
            most_popular_class = item

    return most_popular_class

def knn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''    
    nearest = k_nearest(x,points,k)

    return vote(point_targets[nearest],classes)

def knn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:

    ret_arr = np.zeros(points.shape[0], dtype=int)
    for i in range(points.shape[0]):
        ret_arr[i] = knn(points[i],help.remove_one(points, i),help.remove_one(point_targets, i),classes,k)

    return ret_arr

def knn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:
    guess = knn_predict(points,point_targets,classes,k)
    real = point_targets
    correct = 0
    for i in range(len(point_targets)):
        if guess[i] == real[i]:
            correct += 1

    return correct/len(point_targets)

def knn_confusion_matrix(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    x = y = len(classes)
    matrix = np.zeros((x,y))
    guess = knn_predict(points,point_targets,classes,k)
    target = point_targets
    for n in range(len(target)):
        matrix[guess[n]][target[n]] += 1

    return matrix

def best_k(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
) -> int:
    k_list = np.linspace(1,points.shape[0]-1,points.shape[0]-1,dtype=int)
    max_acc = 0
    max_k = 0
    for item in k_list:
        acc = knn_accuracy(points,point_targets,classes,item)
        if acc > max_acc:
            max_acc = acc
            max_k = item

    return max_k

def knn_plot_points(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
):
    colors = ['yellow','purple','blue']
    guess = knn_predict(points,point_targets,classes,k)

    for i in range(points.shape[0]):
        [x, y] = points[i,:2]

        if guess[i] == point_targets[i]:
            color = 'green'
        else:
            color = 'red'

        plt.scatter(x, y, c=colors[point_targets[i]], edgecolors=color,linewidths=2)

    plt.title('Yellow=0, Purple=1, Blue=2')
    plt.show()

def weighted_vote(
    targets: np.ndarray,
    distances: np.ndarray,
    classes: list
) -> int:
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    # Remove if you don't go for independent section

    result = np.zeros(len(classes))
    dist_sum = 0
    for i in range(targets.shape[0]):
        if distances[i] > 0: # To avoid division by zero
            """Not sure how to handle case when distance is zero."""
            result[targets[i]] += 1/distances[i]
            dist_sum += 1/distances[i]

    return np.argmax(result/dist_sum)

def wknn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    # Remove if you don't go for independent section
    nearest = k_nearest(x,points,k)
    distances = euclidian_distances(x,points[nearest])

    return weighted_vote(point_targets[nearest],distances,classes)


def wknn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    # Remove if you don't go for independent section

    ret_arr = np.zeros(points.shape[0], dtype=int)
    for i in range(points.shape[0]):
        ret_arr[i] = wknn(points[i],help.remove_one(points, i),help.remove_one(point_targets, i),classes,k)
        
    return ret_arr
    


def compare_knns(
    points: np.ndarray,
    targets: np.ndarray,
    classes: list
):
    # Remove if you don't go for independent section

    k = points.shape[0]
    accuracy_knn = []
    accuracy_wknn = []

    for i in range(1,k):
        accuracy_knn.append(knn_accuracy(points,targets,classes,i))
        guess = wknn_predict(points,targets,classes,i)
        real = targets
        correct = 0
        for j in range(len(targets)):
            if guess[j] == real[j]:
                correct += 1

        accuracy_wknn.append(correct/len(targets))
    
    plt.plot(range(1,len(points)),accuracy_knn,c='blue', label='knn')
    plt.plot(range(1,len(points)),accuracy_wknn,c='orange',label='wknn')
    plt.legend()
    plt.show()
                
"""
    B. Theoretical:
        What explains this difference in accuracy between kNN and wkNN when k increases?

    When k increases in kNN, it only takes note of how many classes it finds, and chooses
    the class that has the highest occurance. But this in turn leads to the fact that the kNN
    algorithm can reach clusters that are of the incorrect type, and therefore choose that type.

    However in the wkNN, the algorithm weighs the votes, so when wkNN eventually reaches other clusters
    that are far away, since wkNN weighs its votes, it will give them a lower rating than the ones
    that are close to eachother.

    In comparison, if kNN reaches a high-density cluster far away, it will choose that class for its vote, which in incorrect
    But if wkNN reaches the same high-density cluser far away, it will give those points a lower weight and therefore
    will choose the closer points as its vote class.
"""

d, t, classes = load_iris()
(d_train, t_train), (d_test, t_test) = split_train_test(d, t, train_ratio=0.8)
compare_knns(d_test, t_test, classes)