# Author: Bjarki Laxdal 
# Date: 12.10.22
# Project: 10_boosting
# Acknowledgements: This project was done in collaboration with Gunnlaug Margret, Aron Ingi, Aisha Regina, Steinunn Bjorg, Fanney Einarsdottir, Iris Fridriksdottir
#
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, RandomizedSearchCV)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score, precision_score)

from tools import get_titanic, build_kaggle_submission


def get_better_titanic():
    '''
    Loads the cleaned titanic dataset but change
    how we handle the age column.
    '''

    # Load in the raw data
    # check if data directory exists for Mimir submissions
    # DO NOT REMOVE
    if os.path.exists('./data/train.csv'):
        train = pd.read_csv('./data/train.csv')
        test = pd.read_csv('./data/test.csv')
    else:
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')

    # Concatenate the train and test set into a single dataframe
    # we drop the `Survived` column from the train set
    X_full = pd.concat([train.drop('Survived', axis=1), test], axis=0)
    # print(X_full[:50])

    # The cabin category consist of a letter and a number.
    # We can divide the cabin category by extracting the first
    # letter and use that to create a new category. So before we
    # drop the `Cabin` column we extract these values
    X_full['Cabin_mapped'] = X_full['Cabin'].astype(str).str[0]

    # Then we transform the letters into numbers
    cabin_dict = {k: i for i, k in enumerate(X_full.Cabin_mapped.unique())}
    X_full.loc[:, 'Cabin_mapped'] =\
        X_full.loc[:, 'Cabin_mapped'].map(cabin_dict)

    # We drop multiple columns that contain a lot of NaN values
    # in this assignment
    # Maybe we should
    # CHANGE: removed 'Age' from drop
    X_full.drop(
        ['PassengerId', 'Cabin', 'Name', 'Ticket'],
        inplace=True, axis=1)

    # Instead of dropping the fare column we replace NaN values
    # with the 3rd class passenger fare mean.
    fare_mean = X_full[X_full.Pclass == 3].Fare.mean()
    X_full['Fare'].fillna(fare_mean, inplace=True)

    # Instead of dropping the Embarked column we replace NaN values
    # with `S` denoting Southampton, the most common embarking
    # location
    X_full['Embarked'].fillna('S', inplace=True)


    # CHANGE: replace NaN age values with mean age values of that ticket class

    # X_full['Age'].fillna(round(X_full['Age'].mean(),1), inplace=True)

    mean_age_class_1 = round(X_full[X_full.Pclass == 1]['Age'].mean(),1)
    mean_age_class_2 = round(X_full[X_full.Pclass == 2]['Age'].mean(),1)
    mean_age_class_3 = round(X_full[X_full.Pclass == 3]['Age'].mean(),1)

    X_full['Age'] = X_full.apply(
        lambda row: mean_age_class_1 if np.isnan(row['Age']) and row['Pclass'] == 1 else mean_age_class_2 if np.isnan(row['Age']) and row['Pclass'] == 2 else mean_age_class_3 if np.isnan(row['Age']) and row['Pclass'] == 3 else row['Age'],
        axis=1
    )

    # We then use the get_dummies function to transform text
    # and non-numerical values into binary categories.
    X_dummies = pd.get_dummies(
        X_full,
        columns=['Sex', 'Cabin_mapped', 'Embarked'],
        drop_first=True)

    # We now have the cleaned data we can use in the assignment
    X = X_dummies[:len(train)]
    submission_X = X_dummies[len(train):]
    y = train.Survived
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=5, stratify=y)

    return (X_train, y_train), (X_test, y_test), submission_X


def rfc_train_test(X_train, t_train, X_test, t_test):
    '''
    Train a random forest classifier on (X_train, t_train)
    and evaluate it on (X_test, t_test)
    '''
    classifier = RandomForestClassifier() 
    classifier.fit(X_train,t_train)
    predict = classifier.predict(X_test)

    return (accuracy_score(t_test, predict), precision_score(t_test, predict), recall_score(t_test, predict))


def gb_train_test(X_train, t_train, X_test, t_test):
    '''
    Train a Gradient boosting classifier on (X_train, t_train)
    and evaluate it on (X_test, t_test)
    '''
    classifier = GradientBoostingClassifier() 
    classifier.fit(X_train,t_train)
    predict = classifier.predict(X_test)

    return (accuracy_score(t_test, predict), precision_score(t_test, predict), recall_score(t_test, predict))


def param_search(X, y):
    '''
    Perform randomized parameter search on the
    gradient boosting classifier on the dataset (X, y)
    '''
    # Create the parameter grid
    gb_param_grid = {
        'n_estimators': [x for x in range(1,101)],
        'max_depth': [x for x in range(1,51)],
        'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]}
    # Instantiate the regressor
    gb = GradientBoostingClassifier()
    # Perform random search
    gb_random = RandomizedSearchCV(
        param_distributions=gb_param_grid,
        estimator=gb,
        scoring="accuracy",
        verbose=0,
        n_iter=50,
        cv=4)
    # Fit randomized_mse to the data
    gb_random.fit(X, y)
    # Print the best parameters and lowest RMSE
    return gb_random.best_params_


def gb_optimized_train_test(X_train, t_train, X_test, t_test):
    '''
    Train a gradient boosting classifier on (X_train, t_train)
    and evaluate it on (X_test, t_test) with
    your own optimized parameters
    '''
    # params = param_search(X_train, t_train)
    # print(params)
    # classifier = GradientBoostingClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'],learning_rate=params['learning_rate']) 
    classifier = GradientBoostingClassifier(n_estimators=46, max_depth=5,learning_rate=0.1) 
    classifier.fit(X_train,t_train)
    predict = classifier.predict(X_test)

    return (accuracy_score(t_test, predict), precision_score(t_test, predict), recall_score(t_test, predict))



def _create_submission():
    '''Create your kaggle submission
    '''
    (tr_X, tr_y), (tst_X, tst_y), submission_X = get_better_titanic()
    classifier = GradientBoostingClassifier(n_estimators=46, max_depth=5,learning_rate=0.1) 
    classifier.fit(tr_X,tr_y)
    prediction = classifier.predict(submission_X)
    build_kaggle_submission(prediction)

# if __name__ == "__main__":
    # (tr_X, tr_y), (tst_X, tst_y), submission_X = get_better_titanic()
    # get the first 1 row in the training features
    # print(tr_X[:10])
    # print(rfc_train_test(tr_X, tr_y, tst_X, tst_y))
    # print(gb_train_test(tr_X, tr_y, tst_X, tst_y))
    # print(param_search(tr_X, tr_y)) 

    #{'n_estimators': 46, 'max_depth': 5, 'learning_rate': 0.1} #VIRKAR



    # (tr_X, tr_y), (tst_X, tst_y), submission_X = get_better_titanic()
    # print(gb_train_test(tr_X, tr_y, tst_X, tst_y))
    # (tr_X, tr_y), (tst_X, tst_y), submission_X = get_better_titanic()
    # _create_submission()
    # print(gb_optimized_train_test(tr_X, tr_y, tst_X, tst_y))

