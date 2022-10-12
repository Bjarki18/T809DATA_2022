# Author: Bjarki Laxdal 
# Date: 30.09.22
# Project: 08_SVM
# Acknowledgements: This project was done in collaboration with Gunnlaug Margret, Aron Ingi, Aisha Regina, Steinunn Bjorg, Fanney Einarsdottir, Iris Fridriksdottir
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.


from tools import plot_svm_margin, load_cancer
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, precision_score, recall_score

import numpy as np
import matplotlib.pyplot as plt


def _plot_linear_kernel():
    X, t = make_blobs(n_samples = 40, centers = 2)
    clf = svm.SVC(C=1000,kernel="linear")
    clf.fit(X,t)
    plot_svm_margin(clf,X,t)


def _subplot_svm_margin(
    svc,
    X: np.ndarray,
    t: np.ndarray,
    num_plots: int,
    index: int
):
    '''
    Plots the decision boundary and decision margins
    for a dataset of features X and labels t and a support
    vector machine svc.

    Input arguments:
    * svc: An instance of sklearn.svm.SVC: a C-support Vector
    classification model
    * X: [N x f] array of features
    * t: [N] array of target labels
    '''
    # similar to tools.plot_svm_margin but added num_plots and
    # index where num_plots should be the total number of plots
    # and index is the index of the current plot being generated
    plt.subplot(1,num_plots,index)
    plt.scatter(X[:, 0], X[:, 1], c=t, s=30, cmap=plt.cm.Paired)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    Z = svc.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(
        XX, YY, Z,
        colors='k', levels=[-1, 0, 1],
        alpha=0.5, linestyles=['--', '-', '--'])

    # plot support vectors
    ax.scatter(
        svc.support_vectors_[:, 0],
        svc.support_vectors_[:, 1],
        s=100, linewidth=1, facecolors='none', edgecolors='k')





def _compare_gamma():
    X, t = make_blobs(n_samples=40, centers=2, random_state=6)
    plt.figure()
    clf = svm.SVC(C=1000)
    clf.fit(X,t)
    _subplot_svm_margin(clf, X, t, 3, 1)

    clf = svm.SVC(C=1000, gamma = 0.2)
    clf.fit(X,t)
    _subplot_svm_margin(clf, X, t, 3, 2)

    clf = svm.SVC(C=1000, gamma = 2)
    clf.fit(X,t)
    _subplot_svm_margin(clf, X, t, 3, 3)


    plt.show()


def _compare_C():
    X, t = make_blobs(n_samples=40, centers=2, random_state=6)
    plt.figure()

    clf = svm.SVC(C=1000,kernel="linear")
    clf.fit(X,t)
    _subplot_svm_margin(clf, X, t, 5, 1)

    clf = svm.SVC(C=0.5,kernel="linear")
    clf.fit(X,t)
    _subplot_svm_margin(clf, X, t, 5, 2)

    clf = svm.SVC(C=0.3,kernel="linear")
    clf.fit(X,t)
    _subplot_svm_margin(clf, X, t, 5, 3)

    clf = svm.SVC(C=0.05,kernel="linear")
    clf.fit(X,t)
    _subplot_svm_margin(clf, X, t, 5, 4)

    clf = svm.SVC(C=0.0001,kernel="linear")
    clf.fit(X,t)
    _subplot_svm_margin(clf, X, t, 5, 5)


    plt.show()


def train_test_SVM(
    svc,
    X_train: np.ndarray,
    t_train: np.ndarray,
    X_test: np.ndarray,
    t_test: np.ndarray,
):
    '''
    Train a configured SVM on <X_train> and <t_train>
    and then measure accuracy, precision and recall on
    the test set

    This function should return (accuracy, precision, recall)
    '''
    svc.fit(X_train,t_train)
    y = svc.predict(X_test)
    return (accuracy_score(t_test, y), precision_score(t_test, y), recall_score(t_test, y))


def _compare_all():
    (X_train, t_train), (X_test, t_test) = load_cancer()
    svc = svm.SVC(C=1000, kernel="linear")
    out_lin = train_test_SVM(svc, X_train, t_train, X_test, t_test)


    svc = svm.SVC(C=1000, kernel="rbf")
    out_rbf = train_test_SVM(svc, X_train, t_train, X_test, t_test)

    svc = svm.SVC(C=1000, kernel="poly")
    out_poly = train_test_SVM(svc, X_train, t_train, X_test, t_test)

    return out_lin, out_rbf, out_poly



# if __name__ == "__main__":
    # _plot_linear_kernel()
    # _compare_gamma()
    # _compare_C()
    # (X_train, t_train), (X_test, t_test) = load_cancer()
    # svc = svm.SVC(C=1000)

    # out = train_test_SVM(svc, X_train, t_train, X_test, t_test)
    # print(out)


    # for i in _compare_all():
    #     print(i)