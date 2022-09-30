# Author: Bjarki Laxdal - Bjarki18 AT ru.is 
# Date: 13.09.22
# Project: 05_backprop
# Acknowledgements: This project was done in collaboration with Gunnlaug Margret, Aron Ingi, Aisha Regina, Steinunn Bjorg, Fanney Einarsdottir, Ignas
#

from typing import Union
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from tools import load_iris, split_train_test


def sigmoid(x: float) -> float:
    '''
    Calculate the sigmoid of x
    '''
    if (x<-100):
        return 0.0
    else:
        return 1/(1+np.exp(-x))


def d_sigmoid(x: float) -> float:
    '''
    Calculate the derivative of the sigmoid of x.
    '''
    return sigmoid(x)*(1-sigmoid(x))


def perceptron(
    x: np.ndarray,
    w: np.ndarray
) -> Union[float, float]:
    '''
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    '''
    # 5.48 -> aj = sum w_{ji}z_{i}
    # 5.62 -> aj = sum i=0->D w_{ji}^{1}x_{i}
    w_sum = 0
    for i in range(len(x)):
        w_sum += x[i]*w[i]
    return (w_sum, sigmoid(w_sum))


def ffnn(
    x: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Computes the output and hidden layer variables for a
    single hidden layer feed-forward neural network.
    '''

    '''
        feet forward neural network (ffnn)
        Outputs: y is the result of feeding through steps
            z1 and a1 is from hidden layer
            z0 and z0 from input layer
            a2 from output layer
    ''' 

    z0 = np.insert(x,0,1.0)
    y = np.array([])
    z1 = np.array([1.0])
    a1 = np.array([])
    a2 = np.array([])

    for i in range(M):
        a1 = np.append(a1,perceptron(z0,W1[:,i])[0])
        z1 = np.append(z1,perceptron(z0,W1[:,i])[1])

    for i in range(K):
        a2 = np.append(a2,perceptron(z1,W2[:,i])[0])
        y = np.append(y,perceptron(z1,W2[:,i])[1])


    return y, z0, z1, a1, a2

    
def backprop(
    x: np.ndarray,
    target_y: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Perform the backpropagation on given weights W1 and W2
    for the given input pair x, target_y
    '''

    '''
        when ffnn is ready we can do the backpropagation. first thing we do is call the ffnn function
        because we need all the output variables from that function.
        then we go through the backpropogation, calculating each layer etc.
        dE1 and dE2 are weight differences
        start with dE1 and dE2 have them same shape as W1 and W2.
    '''
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
    delta_k = y-target_y
    delta_j = []
    for i in range(M):
        delta_j.append(d_sigmoid(a1[i])*np.sum(W2[i+1]*delta_k))

    dE1 = np.zeros((W1.shape[0],W1.shape[1]))
    dE2 = np.zeros((W2.shape[0],W2.shape[1]))

    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            dE1[i][j] = delta_j[j]*z0[i]

    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            dE2[i][j] = delta_k[j]*z1[i]

    return y, dE1, dE2


def train_nn(
    X_train: np.ndarray,
    t_train: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
    iterations: int,
    eta: float
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Train a network by:
    1. forward propagating an input feature through the network
    2. Calculate the error between the prediction the network
    made and the actual target
    3. Backpropagating the error through the network to adjust
    the weights.
    '''
    
    W1tr = np.zeros((W1.shape[0],W1.shape[1]))
    W2tr = np.zeros((W2.shape[0],W2.shape[1]))
    N = len(X_train)
    misclassification_rate = []
    E_total = []
    guesses = np.zeros(N)
    

    for i in range(iterations):
        dE1_total = np.zeros((W1.shape[0],W1.shape[1]))
        dE2_total = np.zeros((W2.shape[0],W2.shape[1]))
        error_value = 0
        misclass_value = 0
        for j in range(N):
            target_y = np.zeros(K)
            target_y[t_train[j]] = 1.0
            y, dE1, dE2 = backprop(X_train[j],target_y,M,K,W1,W2)

            guesses[j] = np.argmax(y)

            dE1_total += dE1
            dE2_total += dE2
            
            error_value += target_y*np.log(y) + (1-target_y)*np.log(1-y)

            if (guesses[j] != np.argmax(target_y)):
                misclass_value += 1

        W1 -= eta * dE1_total / N
        W2 -= eta * dE2_total / N

        E_total.append(np.sum(-error_value)/N)
        misclassification_rate.append(misclass_value/N)
        
    W1tr = W1
    W2tr = W2

    return W1tr, W2tr, E_total, misclassification_rate, guesses
    

def test_nn(
    X: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> np.ndarray:
    '''
    Return the predictions made by a network for all features
    in the test set X.
    '''
    guesses = []
    for i in X:
        y, _, _, _, _ = ffnn(i,M,K,W1,W2)
        guesses.append(np.argmax(y))

    return np.array(guesses)

def confusion_matrix(
    test_targets: np.ndarray,
    guesses: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    x = y = len(classes)
    matrix = np.zeros((x,y))
    target = test_targets

    for n in range(len(target)):
        matrix[guesses[n]][target[n]] += 1
    return matrix

def accuracy(
    targets: np.ndarray,
    X: np.ndarray, 
    M: int,
    K: int, 
    W1: np.ndarray, 
    W2: np.ndarray
) -> float:
    prediction = test_nn(X,M,K,W1,W2)
    return accuracy_score(targets,prediction)


def plot_error_misclass(
    iterations: int,
    E_total: np.ndarray,
    misclassification_rate: np.ndarray
):
    # plot E_total
    x = [x for x in range(iterations)]
    y = E_total
    plt.plot(x,y)
    plt.xlabel('Iteration')
    plt.ylabel('E_total')
    plt.show()

    # plot misclassifications rate 
    y1 = misclassification_rate
    plt.plot(x,y1)
    plt.xlabel('Iteration')
    plt.ylabel('Misclassifications Rate')
    plt.show()


# if __name__ == "__main__":
    # print(sigmoid(0.5))
    # print(d_sigmoid(0.2))
    # print(perceptron(np.array([1.0, 2.3, 1.9]),np.array([0.2,0.3,0.1])))
    # print(perceptron(np.array([0.2,0.4]),np.array([0.1,0.4])))
    # features, targets, classes = load_iris()
    # (train_features, train_targets), (test_features, test_targets) = \
    # split_train_test(features, targets)

    # np.random.seed(1234)
    # x = [6.3, 2.5, 4.9, 1.5]
    # K = 3 # number of classes
    # M = 10
    # D = 4
    # # Initialize two random weight matrices
    # W1 = 2 * np.random.rand(D + 1, M) - 1
    # W2 = 2 * np.random.rand(M + 1, K) - 1
    # y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
    # print(y)
    # print(z0)
    # print(z1)
    # print(a1)
    # print(a2)
    # print(ffnn(x,M,K,W1,W2))
    # initialize random generator to get predictable results
    # np.random.seed(42)

    # K = 3  # number of classes
    # M = 6
    # D = train_features.shape[1]

    # x = features[0, :]

    # # create one-hot target for the feature
    # target_y = np.zeros(K)
    # target_y[targets[0]] = 1.0

    # # Initialize two random weight matrices
    # W1 = 2 * np.random.rand(D + 1, M) - 1
    # W2 = 2 * np.random.rand(M + 1, K) - 1

    # y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)


    # np.random.seed(23)
    # features, targets, classes = load_iris()
    # (train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets)
    # x = train_features[0, :]
    # K = 3
    # M = 6
    # D = train_features.shape[1]
    # W1 = 2 * np.random.rand(D + 1, M) - 1
    # W2 = 2 * np.random.rand(M + 1, K) - 1
    # W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(train_features[:20, :], train_targets[:20], M, K, W1, W2, 500, 0.1)
    # y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)


    # target_y = np.zeros(K)
    # target_y[targets[0]] = 1.0
    # y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)
    # print(y)
    # print(dE1)
    # print(dE2)
    # print(W1tr)
    # print(W2tr)
    # print(Etotal)
    # print(misclassification_rate)
    # print(last_guesses)



    # acc = accuracy(test_targets,test_features,M,K,W1tr,W2tr)
    # print(acc)

    # matrix = confusion_matrix(test_targets,test_nn(test_features,M,K,W1tr,W2tr),classes)
    # print(matrix)

    # iterations = 500
    # plot_error_misclass(iterations,Etotal,misclassification_rate)