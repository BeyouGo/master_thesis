### Semi supervised learning using mode a cluster labelling

from sklearn.tree import tree

from EncryptedTrafficClassification.Dataset import Dataset
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import time

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
    features.append('Src Port')
    features.append('Dst Port')
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
times_stored = []


for n_clusters in [100,200,500,1000]:

    ## Set Parameters
    repetition = 1
    classes = [12,13]
    trainingset_size = 10
    testset_size = 40
    test_size = 0.1
    best_feature = True

    # n_clusters = 200
    # birch_threshold = 100
    model_name = "agglo"
    verbose = True

    sizes = []
    steps_ = range(1,100)
    mean_accuracy_semi = [0]*len(steps_)
    mean_accuracy_sup = [0]*len(steps_)



    start_timer = time.clock()

    for index3 in range(0,repetition):

        accuracy_semi = []
        accuracy_sup = []
        for trainingset_size in steps_:
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

                Xall, Yall = shuffle(Xall,Yall)

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
                    X_ = dataset_Xs[index][trainingset_size:]
                    Y_ = dataset_Ys[index][trainingset_size:]
                    continue

                X_label = np.concatenate([X_label, dataset_Xs[index][:trainingset_size]])
                Y_label = np.concatenate([Y_label, dataset_Ys[index][:trainingset_size]])
                X_ = np.concatenate([X_, dataset_Xs[index][trainingset_size:]])
                Y_ = np.concatenate([Y_, dataset_Ys[index][trainingset_size:]])
            ############################################################




            X_unlabel, X_test, Y_unlabel, Y_test = train_test_split(X_, Y_, test_size=test_size, shuffle=True)
            print("Y_label :", Y_label)
            print("Y_label size :", len(Y_label))
            print("Y_label counts:",np.bincount(Y_label))

            ############################################# Choise of Clustering model

            # model = KMeans(n_clusters=n_clusters)
            model = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters) #'ward', 'average', 'complete', 'single'
            # model = Birch(branching_factor=50,n_clusters=n_clusters)#,compute_labels=True,threshold=1000

            X_mix = np.concatenate([X_label, X_unlabel])
            model.fit(X_mix)

            ############################################# Create cluster_population
            labels_ = model.labels_

            clusters_population = {}
            cluster_datasets = {}
            for index in range(0, len(X_label)):
                cluster_numb = model.labels_[index]
                true_label = Y_label[index]

                if not cluster_numb in clusters_population.keys():
                    clusters_population[cluster_numb] = []
                    clusters_population[cluster_numb].append(true_label)

                    cluster_datasets[cluster_numb] = {}
                    cluster_datasets[cluster_numb][0] = []
                    cluster_datasets[cluster_numb][0].append(X_label[index])
                    cluster_datasets[cluster_numb][1] = []
                    cluster_datasets[cluster_numb][1].append(Y_label[index])

                else:
                    clusters_population[cluster_numb].append(true_label)

                    cluster_datasets[cluster_numb][0].append(X_label[index])
                    cluster_datasets[cluster_numb][1].append(Y_label[index])

            for index in range(len(Y_label), len(X_mix)):

                cluster_num = model.labels_[index]
                if not cluster_num in cluster_datasets.keys():
                    continue
                if not (2 in cluster_datasets[cluster_num].keys()) or ( 3 in cluster_datasets[cluster_num].keys() ) :
                    cluster_datasets[cluster_num][2] = []
                    cluster_datasets[cluster_num][2].append(X_mix[index])
                    cluster_datasets[cluster_num][3] = []
                    cluster_datasets[cluster_num][3].append(index)
                else:
                    cluster_datasets[cluster_num][2].append(X_mix[index])
                    cluster_datasets[cluster_num][3].append(index)


            total_cluster = {}
            for index in range(0, len(X_mix)):
                cluster_numb = model.labels_[index]

                if not cluster_numb in total_cluster.keys():
                    total_cluster[cluster_numb] = [1, 0]

                    if index < len(X_label):
                        total_cluster[cluster_numb][1] = 1

                else:
                    total_cluster[cluster_numb][0] += 1
                    if index < len(X_label):
                        total_cluster[cluster_numb][1] += 1

            total_label = 0
            cluster_avec_label = 0
            total_avec_label = 0
            total_sans_label = 0
            cluster_sans_label = 0
            for cluster_numb in total_cluster.keys():
                total = total_cluster[cluster_numb][0]
                n_labeled_element = total_cluster[cluster_numb][1]
                total_label += n_labeled_element
                if n_labeled_element > 0:
                    cluster_avec_label += 1
                    total_avec_label += total
                else:
                    cluster_sans_label += 1
                    total_sans_label += total

            print("number of labeled sample", len(X_label))
            print("cluster avec label: ", cluster_avec_label, " (", total_avec_label, " elements )")
            print("cluter sans label: ", cluster_sans_label, " (", total_sans_label, " elements )")


            estimations = []

            clusters = {}

            for cluster_name in clusters_population.keys():
                population = clusters_population[cluster_name]
                counts = np.bincount(population)

                estimation = len(population) - (float(len(population) - counts[np.argmax(counts)]))
                estimations.append(estimation)
                clusters[cluster_name] = np.argmax(counts)

            X_semisup = []
            Y_semisup = []

            X = X_label
            passed = 0
            for index in range(0, len(model.labels_)):
                if not model.labels_[index] in clusters.keys():
                    passed = passed + 1
                    continue

                if (index >= len(X_label)):
                    Y_semisup.append(clusters[labels_[index]])
                    relative_index = index - len(X_label)
                    X_semisup.append(X_unlabel[relative_index])
                else:
                    Y_semisup.append(Y_label[index])
                    X_semisup.append(X_label[index])


            ################### Supervised [Labeled]###################################################

            print("############## Supervised [Labeled]########")
            basic_model = RandomForestClassifier(n_estimators=100)
            basic_model.fit(X_label, Y_label)
            pred_lab_test = basic_model.predict(X_test)
            test_accuracy = np.mean(pred_lab_test.ravel() == Y_test.ravel())

            print("trained on",len(Y_label),"samples")
            print("Accuracy :",test_accuracy)
            del basic_model

            accuracy_sup.append(test_accuracy)

            ###################### Semi - Supervised #############
            print("############## Semi - Supervised ########")
            semi_supervised_model = RandomForestClassifier(n_estimators=100)
            semi_supervised_model.fit(X_semisup, Y_semisup)
            pred_lab_test = semi_supervised_model.predict(X_test)
            test_accuracy = np.mean(pred_lab_test.ravel() == Y_test.ravel())
            del semi_supervised_model
            # if verbose :
            print("trained on",len(Y_semisup),"samples")
            print("Accuracy :",test_accuracy)
            accuracy_semi.append(test_accuracy)
            # confmat = confusion_matrix(Y_test.ravel(), pred_lab_test.ravel())

            if index3 == 0:
                sizes.append(trainingset_size)
        mean_accuracy_semi = np.sum([mean_accuracy_semi, accuracy_semi], axis=0)
        mean_accuracy_sup = np.sum([mean_accuracy_sup, accuracy_sup], axis=0)

    end = time.clock()
    print( "time : ", (end - start_timer)/(repetition*100) )
    times_stored.append((end - start_timer)/(repetition*100))

    plt.figure(figsize=(15,5))

    plt.plot(sizes,(mean_accuracy_semi/repetition)*100)
    plt.plot(sizes,(mean_accuracy_sup/repetition)*100)
    plt.ylim((80,102))
    plt.legend(["semi-supervised","supervised"])

    # plt.title(graph_title)
    plt.xlabel("Nbr of training samples per class",fontsize=18)
    plt.ylabel("Accuracy [%]", fontsize=18)
    plt.tight_layout()
    fname = "../figures_saved/"+str(model_name)+"_nosub_"+str(n_clusters)+".png"
    # plt.savefig(fname)
    # plt.show()

print(times_stored)