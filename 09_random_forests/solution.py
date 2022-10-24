# Author: Bjarki Laxdal 
# Date: 12.10.22
# Project: 09_random_forests
# Acknowledgements: This project was done in collaboration with Gunnlaug Margret, Aron Ingi, Aisha Regina, Steinunn Bjorg, Fanney Einarsdottir, Iris Fridriksdottir
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict, train_test_split, cross_val_score
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score, precision_score)

from collections import OrderedDict

from torch import DeserializationStorageContext


class CancerClassifier:
    '''
    A general class to try out different sklearn classifiers
    on the cancer dataset
    '''
    def __init__(self, classifier, train_ratio: float = 0.7):
        self.classifier = classifier
        cancer = load_breast_cancer()
        self.X = cancer.data  # all feature vectors
        self.t = cancer.target  # all corresponding labels
        self.X_train, self.X_test, self.t_train, self.t_test =\
            train_test_split(
                cancer.data, cancer.target,
                test_size=1-train_ratio, random_state=109)

        # Fit the classifier to the training data here
        self.classifier.fit(self.X_train, self.t_train)
        self.prediction = self.classifier.predict(self.X_test)

    def confusion_matrix(self) -> np.ndarray:
        '''Returns the confusion matrix on the test data
        '''
        return confusion_matrix(self.t_test, self.prediction)

    def accuracy(self) -> float:
        '''Returns the accuracy on the test data
        '''
        return accuracy_score(self.t_test, self.prediction)

    def precision(self) -> float:
        '''Returns the precision on the test data
        '''
        return precision_score(self.t_test, self.prediction)

    def recall(self) -> float:
        '''Returns the recall on the test data
        '''
        return recall_score(self.t_test, self.prediction)

    def cross_validation_accuracy(self) -> float:
        '''Returns the average 10-fold cross validation
        accuracy on the entire dataset.
        '''
        return np.mean(cross_val_score(self.classifier, self.X, self.t, cv=10))

    def feature_importance(self) -> list:
        '''
        Draw and show a barplot of feature importances
        for the current classifier and return a list of
        indices, sorted by feature importance (high to low).
        '''
        ...
        importance = self.classifier.feature_importances_
        plt.bar(range(len(importance)),importance)
        plt.xlabel("Feature index")
        plt.ylabel("Features importance")  
        plt.show()
        
        indx = np.argsort(importance)
        return np.flip(indx)

    # def test(self):
    #     pred = cross_val_predict(self.classifier, self.X, self.t, cv=10)
    #     print(len(self.X))
    #     print(len(self.t))
    #     print(np.mean(cross_val_score(self.classifier, self.X, self.t, cv=10,scoring='recall')))
    #     print(np.mean(cross_val_score(self.classifier, self.X, self.t, cv=10,scoring='precision')))
    #     return confusion_matrix(self.t,pred)

def _plot_oob_error():
    RANDOM_STATE = 1337
    ensemble_clfs = [
        ("RandomForestClassifier, max_features='sqrt'",
            RandomForestClassifier(
                n_estimators=100,
                warm_start=True,
                oob_score=True,
                max_features="sqrt",
                random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features='log2'",
            RandomForestClassifier(
                n_estimators=100,
                warm_start=True,
                max_features='log2',
                oob_score=True,
                random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features=None",
            RandomForestClassifier(
                n_estimators=100,
                warm_start=True,
                max_features=None,
                oob_score=True,
                random_state=RANDOM_STATE))]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    min_estimators = 30
    max_estimators = 175

    cancer = load_breast_cancer()

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(cancer.data,  cancer.target)  # Use cancer data here
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()


def _plot_extreme_oob_error():
    RANDOM_STATE = 1337
    ensemble_clfs = [
        ("ExtraTreesClassifier, max_features='sqrt'",
            ExtraTreesClassifier(
                n_estimators=100,
                warm_start=True,
                oob_score=True,
                max_features="sqrt",
                bootstrap=True,
                random_state=RANDOM_STATE)),
        ("ExtraTreesClassifier, max_features='log2'",
            ExtraTreesClassifier(
                n_estimators=100,
                warm_start=True,
                max_features='log2',
                oob_score=True,
                bootstrap=True,
                random_state=RANDOM_STATE)),
        ("ExtraTreesClassifier, max_features=None",
            ExtraTreesClassifier(
                n_estimators=100,
                warm_start=True,
                max_features=None,
                oob_score=True,
                bootstrap=True,
                random_state=RANDOM_STATE))]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    min_estimators = 30
    max_estimators = 175

    cancer = load_breast_cancer()

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(cancer.data,  cancer.target)  # Use cancer data here
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    # part 1.2 
    classifier_type = DecisionTreeClassifier()
    cc = CancerClassifier(classifier_type)
    # print(cc.confusion_matrix())
    # print(cc.accuracy())
    # print(cc.precision())
    # print(cc.recall())
    # print(cc.cross_validation_accuracy())
    # print(cc.test())

    # part 2.1

    # classifier_type = RandomForestClassifier()
    # cc = CancerClassifier(classifier_type)
    # print(cc.confusion_matrix())
    # print(cc.accuracy())
    # print(cc.precision())
    # print(cc.recall())
    # print(cc.cross_validation_accuracy())

    # max_acc = 0
    # best_feat_max = 0
    # best_tree = 0

    # for feat in [i for i in range(1,50,5)]:
    #     for tree in [i for i in range(1,200,25)]:
    #         classifier_type = RandomForestClassifier(max_features = feat, n_estimators = tree)
    #         cc = CancerClassifier(classifier_type)
    #         temp_acc = cc.cross_validation_accuracy()
    #         if (temp_acc > max_acc):
    #             max_acc = temp_acc
    #             best_feat_max = feat
    #             best_tree = tree
    #     print(feat)

    # print(f'max_acc = {max_acc}') #0.9666666666666668
    # print(f'best_feat_max = {best_feat_max}') #16
    # print(f'best_tree = {best_tree}') #126

    # classifier_type = RandomForestClassifier(max_features=16, n_estimators=126) 
    # cc = CancerClassifier(classifier_type)
    # print(cc.confusion_matrix())
    # print(cc.accuracy())
    # print(cc.precision())
    # print(cc.recall())
    # print(cc.cross_validation_accuracy())

    # part 2.2
    # classifier_type = RandomForestClassifier()
    # cc = CancerClassifier(classifier_type)
    # feature_idx = cc.feature_importance()
    # print(feature_idx)

    #part 2.4
    
    # _plot_oob_error()

    #part 3.1
    # classifier_type = ExtraTreesClassifier()
    # cc = CancerClassifier(classifier_type)
    # print(cc.confusion_matrix())
    # print(cc.accuracy())
    # print(cc.precision())
    # print(cc.recall())
    # print(cc.cross_validation_accuracy())
    # feature_idx = cc.feature_importance()
    # #part 3.2
    # _plot_extreme_oob_error()

