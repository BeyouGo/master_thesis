##################################################################################################
######################### Scenario 4: Unknown classes ###########################
##################################################################################################
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


def get_model():
    model = RandomForestClassifier(n_estimators=20, random_state=0, criterion='entropy')
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
known_classes = [0, 2, 4, 6, 10]
Xall, Yall = dataset.get_balanced_data([0, 2, 4, 6, 10, 1, 3, 5], features)
Yall[Yall == 1] = 15
Yall[Yall == 3] = 15
Yall[Yall == 5] = 15

### Xall and Yall contain the flows of all the files
test_size = 0.2
target = np.unique(Yall)
print(len(Yall))
X_train, X_test, Y_train, Y_test = train_test_split(Xall, Yall, test_size=test_size,
                                                    random_state=random.randint(1000))

X_unknown, Y_unknown = dataset.get_balanced_data([7, 8, 9, 11, 12, 13], features)

Y_unknown = [15] * len(Y_unknown)

X_test = np.concatenate([X_test, X_unknown])
Y_test = np.concatenate([Y_test, Y_unknown])

X_trash, X_test, Y_trash, Y_test = train_test_split(X_test, Y_test, test_size=0.99, random_state=random.randint(1000),shuffle=True)

model = get_model()
model.fit(X_train, Y_train)

pred_lab_test = model.predict(X_test)
test_accuracy = np.mean(pred_lab_test.ravel() == Y_test.ravel())



print("trained on", len(Y_test), "samples")
print("Accuracy :", test_accuracy)
print(pred_lab_test.ravel()[:20])
print(Y_test.ravel()[:20])

confmat = confusion_matrix(Y_test.ravel(),pred_lab_test.ravel())

print(confmat)

np.savetxt('confmat_scenario9.csv',
           confmat,
           delimiter=',', fmt="%d"
          )