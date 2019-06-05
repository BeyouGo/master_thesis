
from EncryptedTrafficClassification.Dataset import Dataset
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, estimate_bandwidth, Birch
from sklearn.cluster import AffinityPropagation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

dataset = Dataset("..\\flows\\")
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

def get_best_features():
    features = []
    features.append('Pkt Len Max')
    features.append('Flow IAT Max')
    # features.append('Src Port')
    # features.append('Dst Port')
    return features

def get_all_features():
    # All Features
    features = []
    for feature in featureNames:

        if feature in ['Flow ID', 'Src IP', 'Dst IP', 'Protocol',
                       'Timestamp', 'Label', 'Src Port', 'Dst Port']:
            continue

        features.append(feature)
    return features



repetition = 10
classes = [12,13]
trainingset_size = 10
testset_size = 500
test_size = 0.2
best_feature = True
birch_threshold = 100000
n_clusters = 200
model_name = "Birch with Decision Tree sub-clustering"
graph_title = model_name +" ( threshold = " + str(birch_threshold)+" )"
## discard cluster with less than 5 elements, decision tree subclustering,

semi_better = 0
total_ = 0

cluster_sizes = [10,20,30,40,50,60,70,80,90,100]
mean_accuracies = [0] * len(cluster_sizes)
mean_sup_accuracies = [] * len(cluster_sizes)
mean_sizes = [0] * len(cluster_sizes)



print("#####################################################################")
print("############### step  - ", trainingset_size, "##############")
print("#####################################################################")
## Get features set

if best_feature:
    features = get_best_features()
else:
    features = get_all_features()

## Dataset Creation

dataset_Xs = {}
dataset_Ys = {}
for class_index in classes:
    Xall, Yall = dataset.get_balanced_data([class_index], features)

    Xall = np.array(Xall)
    Yall = np.array(Yall)

    Xall, Yall = shuffle(Xall, Yall)

    dataset_Xs[class_index] = Xall
    dataset_Ys[class_index] = Yall

X_label = []
Y_label = []
X_ = []
Y_ = []

for index in classes:

    if len(Y_label) == 0:
        X_label = dataset_Xs[index][:trainingset_size]
        Y_label = dataset_Ys[index][:trainingset_size]
        X_ = dataset_Xs[index][trainingset_size:int(trainingset_size + testset_size)]
        Y_ = dataset_Ys[index][trainingset_size:int(trainingset_size + testset_size)]
        continue

    X_label = np.concatenate([X_label, dataset_Xs[index][:trainingset_size]])
    Y_label = np.concatenate([Y_label, dataset_Ys[index][:trainingset_size]])
    X_ = np.concatenate([X_, dataset_Xs[index][trainingset_size:int(trainingset_size + testset_size)]])
    Y_ = np.concatenate([Y_, dataset_Ys[index][trainingset_size:int(trainingset_size + testset_size)]])
############################################################

X_, Y_ = shuffle(X_, Y_)

## add new class
X_newclass, y_newclass = dataset.getData([11], features)

X_ = np.concatenate([X_, X_newclass[:testset_size]])
Y_ = np.concatenate([Y_, y_newclass[:testset_size]])
X_, Y_ = shuffle(X_, Y_)

####


X_unlabel, X_test, Y_unlabel, Y_test = train_test_split(X_, Y_, test_size=test_size, shuffle=True)
# print("Y_label :", Y_label)
print("Y_label size :", len(Y_label))
print("Y_label counts:", np.bincount(Y_label))

############################################# Choise of Clustering model

model = KMeans(n_clusters=n_clusters)
# model = AffinityPropagation()
# model = AgglomerativeClustering(linkage='average',n_clusters=n_clusters) #'ward', 'average', 'complete', 'single'
# model = Birch(branching_factor=50, n_clusters=None, threshold=birch_threshold)#,compute_labels=True
X_mix = np.concatenate([X_label, X_unlabel])
model.fit(X_mix)

############################################# Create cluster_population

labels_ = model.labels_

print(np.bincount(labels_))

clusters = {}

for index in range(0, len(labels_)):

    labelled = index < len(X_label)
    cluster_id = labels_[index]

    if labelled: real_label = Y_label[index]

    # print(cluster_id, clusters)
    if not cluster_id in clusters.keys():
        clusters[cluster_id] = {}
        clusters[cluster_id][0] = []  # X_label
        clusters[cluster_id][1] = []  # y_label
        clusters[cluster_id][2] = []  # X_unlabel
        clusters[cluster_id][3] = []  # index

        clusters[cluster_id][3].append(index)

        if labelled:
            clusters[cluster_id][0].append(X_label[index])
            clusters[cluster_id][1].append(Y_label[index])
        else:
            clusters[cluster_id][2].append(X_mix[index])

    else:
        clusters[cluster_id][3].append(index)
        if labelled:
            clusters[cluster_id][0].append(X_label[index])
            clusters[cluster_id][1].append(Y_label[index])
        else:
            clusters[cluster_id][2].append(X_mix[index])

new_class_name = 11
new_class_list = []

y_mix_label = [-1] * len(X_mix)

for index in range(0, len(clusters)):

    if not index in clusters.keys():
        continue

    X_c_label = clusters[index][0]
    y_c_label = clusters[index][1]
    X_c_unlabel = clusters[index][2]
    sample_indexes = clusters[index][3]

    # if len(X_c_unlabel) > 80 and len(y_c_label) < 3:
    #     print("###### NEW CLASS ", new_class_name)
    #     for index2 in range(0, len(sample_indexes)):
    #         sample_index = sample_indexes[index2]
    #         y_mix_label[sample_index] = new_class_name
    #     # new_class_list.append(new_class_name)
    #     # new_class_name += 5
    #     continue

    if len(y_c_label) > 0:
        counts = np.bincount(y_c_label)
        guessed_label = np.argmax(counts)

        for index2 in range(0, len(sample_indexes)):
            # if sample_index >= len(y_c_label):
            sample_index = sample_indexes[index2]
            y_mix_label[sample_index] = guessed_label

X_semisup = []
Y_semisup = []

X = X_label

for index in range(0, len(y_mix_label)):

    if index < len(Y_label):
        Y_semisup.append(Y_label[index])
        X_semisup.append(X_label[index])

    if y_mix_label[index] == -1:
        continue

    Y_semisup.append(y_mix_label[index])
    X_semisup.append(X_mix[index])

################### Supervised [Labeled]###################################################

print("############## Supervised [Labeled]########")
print(np.unique(Y_label))
basic_model = RandomForestClassifier(n_estimators=100)
basic_model.fit(X_label, Y_label)
pred_lab_test = basic_model.predict(X_test)
test_accuracy_sup = np.mean(pred_lab_test.ravel() == Y_test.ravel())

print("trained on", len(Y_label), "samples")
print("Accuracy :", test_accuracy_sup)
print(pred_lab_test.ravel()[:20])
print(Y_test.ravel()[:20])
# print()

###################### Semi - Supervised #############
print("############## Semi - Supervised ########")

print(np.unique(Y_semisup))
print(np.bincount(Y_semisup))

supervised_model = RandomForestClassifier(n_estimators=100)
supervised_model.fit(X_semisup, Y_semisup)
pred_lab_test = supervised_model.predict(X_test)
test_accuracy_semi = np.mean(pred_lab_test.ravel() == Y_test.ravel())
proba = supervised_model.predict_proba(X_test[:10])

print("trained on", len(Y_semisup), "samples")
print("Accuracy :", test_accuracy_semi)
print(pred_lab_test.ravel()[:20])
print(Y_test.ravel()[:20])
