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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from EncryptedTrafficClassification.data_helper import LoadData
from EncryptedTrafficClassification.Dataset import Dataset
import itertools

########################################################################################
#######################     Feature Selection Algorithm     ############################
########################################################################################

## Relate to chapter 4.1

featureNames = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol',
                'Timestamp', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
                'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
                'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
                'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
                'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
                'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot',
                'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
                'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
                'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
                'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s',
                'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
                'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt',
                'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
                'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
                'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg',
                'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg',
                'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts',
                'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
                'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
                'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
                'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label']

dataset = Dataset("..\\flows\\")
x = range(0, 14)
permutations = list(itertools.combinations(x, 2))


def get_all_features():
    # All Features
    features = []
    for feature in featureNames:

        if feature in ['Flow ID', 'Src IP', 'Dst IP', 'Protocol',
                       'Timestamp', 'Label','Src Port','Dst Port']:
            continue

        features.append(feature)
    return features

def get_best_features():
    features = []
    features.append('Flow Duration')
    # features.append('Tot Fwd Pkts')
    # features.append('Fwd Pkt Len Mean')
    # features.append('Flow Pkts/s')
    return features


results = []
for permutation in permutations:

    if permutation[0] == permutation[1]:
        continue
    binary_classes = permutation

    # features = get_all_features()
    features = get_best_features()

    Xall, Yall = dataset.get_balanced_data(binary_classes,features)

    Xall = np.array(Xall)
    Yall = np.array(Yall)

    ### Xall and Yall contain the flows of all the files
    test_size = 0.3
    seed = random.randint(1000)
    X_train, X_test, Y_train, Y_test = train_test_split(Xall, Yall, test_size=test_size, random_state=seed)

    model = RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=5, max_features="auto")

    training_start = time.time()

    ### Train the selected model
    model.fit(X_train, Y_train)

    training_end = time.time()

    ### Compute score using test set
    result = model.score(X_test, Y_test)

    pred_lab_test = model.predict(X_test)
    test_accuracy = np.mean(pred_lab_test.ravel() == Y_test.ravel())

    print("Size: ",len(Yall)/2," --- ",binary_classes, " :" , str(test_accuracy))
    results.append(test_accuracy)

print("Accuracy :",np.sum(results)/len(permutations))
