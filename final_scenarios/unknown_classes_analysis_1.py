##################################################################################################
#########################          Scenario 4: Unknown classes         ###########################
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
probability_table = []
for threshold in [0.1,0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    print("### Threshold =",threshold)
    probability_line = []
    for recognized_class in range(0, 14):

        all_classes = np.array(range(0, 14))
        selected_classes = np.delete(all_classes, recognized_class)
        Xall, Yall = dataset.get_balanced_data(all_classes, features)

        ### Xall and Yall contain the flows of all the files
        test_size = 0.1
        target = np.unique(Yall)
        X_train, X_test, Y_train, Y_test = train_test_split(Xall, Yall, test_size=test_size,
                                                            random_state=random.randint(1000))

        X_unknown, Y_unknown = dataset.get_balanced_data([recognized_class], features)

        model = get_model()
        model.fit(X_train, Y_train)

        result = model.predict_proba(X_unknown)


        threshold = 0.5
        good = 0
        for res in result:
            if np.max(res) < threshold:
                good = good + 1

        accuracy = float(good) / len(X_unknown)
        print("Class", recognized_class,"-",accuracy)
        probability_line.append(accuracy)
    probability_table.append(probability_line)


print(probability_table)

np.savetxt('scenario8_threshold.csv',
           np.transpose(probability_table),
           delimiter=',',
           fmt="%f"
          )