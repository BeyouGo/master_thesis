##################################################################################################
######################### Scenario 4: Unknown classes ###########################
##################################################################################################
import matplotlib.pyplot as plt
from sklearn.tree import tree
import time
from numpy import *
import numpy as np
from os import listdir
from os.path import isfile, join
from operator import itemgetter
from pandas import read_csv
from pandas.core.sparse import array
from sklearn import svm
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn.mixture import GMM, GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from EncryptedTrafficClassification.Dataset import Dataset
import itertools
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle


def get_model():
    model = RandomForestClassifier(n_estimators=200, random_state=0, criterion='entropy')
    return model


def get_best_features():
    features = []
    features.append('Pkt Len Max')
    features.append('Flow IAT Max')
    # features.append('Src Port')
    # features.append('Dst Port')
    return features


dataset = Dataset("..\\flows\\")

class_names = ['aim', 'email', 'facebook', 'ftps', 'hang', 'icq', 'netflix', 'scp', 'sftp', 'skype', 'spotify', 'vimeo',
               'voip', 'youtube']
################ 0 ##### 1 ######## 2 ####### 3 ##### 4 #### 5 ####### 6 ##### 7 ##### 8 ##### 9 ###### 10 ##### 11 ##### 12 ###### 13 ###
features = get_best_features()
mean_accuracy = [0] * 149
for index1 in range(0, 20):
    accuracies = []
    for trainingset_size in range(1, 150):

        classes = range(0,14)
        # classes = [0, 6]
        # trainingset_size = 2
        testset_size = 40

        dataset_Xs = {}
        dataset_Ys = {}
        for class_index in classes:
            Xall, Yall = dataset.get_balanced_data([class_index], features)
            Xall, Yall = shuffle(Xall,Yall)
            dataset_Xs[class_index] = Xall
            dataset_Ys[class_index] = Yall

        X_train = []
        X_test = []
        Y_train = []
        Y_test = []
        for index in classes:

            if len(Y_train) == 0:
                X_train = dataset_Xs[index][:trainingset_size]
                Y_train = dataset_Ys[index][:trainingset_size]
                X_test = dataset_Xs[index][trainingset_size:trainingset_size + testset_size]
                Y_test = dataset_Ys[index][trainingset_size:trainingset_size + testset_size]
                continue

            X_train = np.concatenate([X_train, dataset_Xs[index][:trainingset_size]])
            Y_train = np.concatenate([Y_train, dataset_Ys[index][:trainingset_size]])

            X_test = np.concatenate([X_test, dataset_Xs[index][trainingset_size:trainingset_size + testset_size]])
            Y_test = np.concatenate([Y_test, dataset_Ys[index][trainingset_size:trainingset_size + testset_size]])

        ### Xall and Yall contain the flows of all the files
        test_size = 0.2

        X_train, Y_train = shuffle(X_train, Y_train)
        X_test, Y_test = shuffle(X_test, Y_test)

        model = get_model()
        model.fit(X_train, Y_train)
        pred_lab_test = model.predict(X_test)
        test_accuracy = np.mean(pred_lab_test.ravel() == Y_test.ravel())
        # print(test_accuracy)
        accuracies.append(test_accuracy)
        del model
    print(accuracies)
    mean_accuracy = np.sum([mean_accuracy,accuracies],axis=0)



plt.plot(np.array(mean_accuracy)/20)
plt.show()
#
# print(np.bincount(Y_train))
# print(np.bincount(Y_test))
# print("trained on", len(Y_train), "samples")
# print("Accuracy :", test_accuracy)
# print(pred_lab_test.ravel()[:20])
# print(Y_test.ravel()[:20])
#
# confmat = confusion_matrix(Y_test.ravel(),pred_lab_test.ravel())
#
# print(confmat)
#
# np.savetxt('confmat_scenario9.csv',
#            confmat,
#            delimiter=',', fmt="%d"
#           )
