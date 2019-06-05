from numpy import *
import numpy as np
from pandas import read_csv
# from pandas.core.sparse import array
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from os import listdir
from os.path import isfile, join


class Dataset:
    def __init__(self, flow_path):

        ## flow location on the computer
        self.flows_path = flow_path
        self.files_list = [f for f in listdir(self.flows_path) if isfile(join(self.flows_path, f))]

        ## All flows and labels
        self.Xall = []
        self.Yall = []

        ## info on features (set in read_csv)
        self.featureNames = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol',
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
        self.featureCount = 0
        self.extract_csv()

    def extract_csv(self):

        # Initialize data
        self.Xall = np.array(1)
        self.Yall = np.array(1)
        first = True

        for filename in self.files_list:
            if (filename[:3] in ['tor', 'vpn']):  ## Tor and Vpn are discared
                continue

            data = read_csv(self.flows_path + '\\' + filename, names=self.featureNames)

            ## we skip labels and flowId, ip source, ip dest, source port, dest port and timestamp !!!!!!!!!!!!!!!!!!!!!
            # X = data.values[1:, 7:len(data.values[1]) - 2]
            X = data.values[1:,:]

            ## Replace non numeric value by zeros

            X[:,len(data.values[1]) - 1] = 0
            X[:,0] = 0
            X[:,1] = 0
            X[:,3] = 0
            X[:,5] = 0
            X[:,6] = 0

            X = X.astype('float64')
            X[np.isnan(X)] = 0
            X[np.isinf(X)] = 0

            # Set Class value
            Y = data.values[1:, len(data.values[1]) - 2]
            Y = Y.astype('float64')




            class_names = ['aim', 'email', 'facebook', 'ftps', 'hang', 'icq', 'netflix', 'scp', 'sftp', 'skype',
                           'spotify', 'vimeo', 'voip', 'youtube']

            for i in range(0, len(class_names)):
                if str(filename).startswith(class_names[i][:3]):
                    Y[:] = i
            X, Y = self.select_samples(X, Y)


            # if str(filename).startswith('aim'):
            #     Y[:] = 0
            # elif str(filename).startswith('email'):
            #     Y[:] = 1
            # elif str(filename).startswith('facebook'):
            #     Y[:] = 2
            # elif str(filename).startswith('ftps'):
            #     Y[:] = 3
            # elif str(filename).startswith('hang'):
            #     Y[:] = 4
            # elif str(filename).startswith('icq'):
            #     Y[:] = 5
            # elif str(filename).startswith('netflix'):
            #     Y[:] = 6
            # elif str(filename).startswith('scp'):
            #     Y[:] = 7
            # elif str(filename).startswith('sftp'):
            #     Y[:] = 8
            # elif str(filename).startswith('skype'):
            #     Y[:] = 9
            # elif str(filename).startswith('spotify'):
            #     Y[:] = 10
            # elif str(filename).startswith('vimeo'):
            #     Y[:] = 11
            # elif str(filename).startswith('voip'):
            #     Y[:] = 12
            # elif str(filename).startswith('youtube'):
            #     Y[:] = 13
            # else:
            #     Y[:] = 14

            ## Concatenate all flows in a single array
            if first:
                self.Xall = X
                self.Yall = Y
                first = False
            else:
                self.Xall = np.concatenate([self.Xall, X])
                self.Yall = np.concatenate([self.Yall, Y])

        return

    def getData(self, classes, selected_features):

        n_class = len(classes)

        ## Select Classes:
        X_classSelected = np.array(1)
        Y_classSelected = np.array(1)

        first = True
        first_class = True

        for class_index in classes:
            X = np.array(self.Xall[self.Yall == class_index])
            Y = np.array(self.Yall[self.Yall == class_index]).astype(int)

            if first_class:
                X_classSelected = X
                Y_classSelected = Y
                first_class = False
            else:
                X_classSelected = np.concatenate([X_classSelected, X])
                Y_classSelected = np.concatenate([Y_classSelected, Y])

        X = X_classSelected
        Y = Y_classSelected

        selected_features_index = []

        for f in selected_features:
            if f in self.featureNames:
                index = self.featureNames.index(f)
                selected_features_index.append(index)

        X_final = []
        Y_final = []
        for i in range(0, len(X)):
            sample = X[i]

            # if sample[2] < 8 and sample[4] < 8: # remove sample with port number below 8
            #     continue

            Y_final.append(Y[i])

            troncated_sample = []
            for feature_index in selected_features_index:
                troncated_sample.append(sample[feature_index])

            # sample = np.split(sample, selected_features_index)
            # print(troncated_sample)
            X_final.append(troncated_sample)

        # X = np.array(X_final)
        # Y = np.array(Y_final)

        X = X_final
        Y = Y_final
        # print(Y)

        # x_features_selected = []
        # for f in selected_features:
        #     if f in self.featureNames:
        #         index = self.featureNames.index(f)
        #         if first:
        #             x_features_selected = X[:, index]
        #             first = False
        #         else:
        #             x_features_selected = np.vstack([x_features_selected, X[:, index]])

        return [X, Y]

    def get_balanced_data(self, classes, selected_features):

        [X, Y] = self.getData(classes, selected_features)

        ## Take the class with the minimum sample
        min_elements = 1000000
        for class_index in classes:
            value = Y.count(class_index)
            if value < min_elements:
                min_elements = value

        X_balanced = np.array(1)
        Y_balanced = np.array(1)
        first = True

        X = np.array(X)
        Y = np.array(Y)
        for class_name in classes:

            X_unique_class = X[Y == class_name]

            if first :
                X_balanced = X_unique_class[:min_elements]
                Y_balanced = [class_name] * min_elements
                first = False

            else:

                X_balanced = np.concatenate([X_balanced, X_unique_class[:min_elements]])
                Y_balanced = np.concatenate([Y_balanced,[class_name] * min_elements])

        # print(Y_balanced)
        # print(len(X_balanced))

        # print("MIN: ", min_elements)
        # for i in range(0, len(Y)):
        # print("hello")

        return [X_balanced, Y_balanced]


    def select_samples(self,X,y):

        # print(y)

        Xnew = []
        ynew = []
        for index in range(0,len(y)):
            currX = X[index]
            curry = y[index]

            feature_index = self.featureNames.index( 'Pkt Len Max')
            if currX[feature_index] < 7:
                continue
            #
            if currX[4] in [53,137,138,161,123]: # DNS Resquest
                continue

            feature_index2 = self.featureNames.index('Pkt Len Max')
            feature_index3 = self.featureNames.index('Flow IAT Max')
            # print(curry,int(currX[2]),"-",int(currX[4]),"-",currX[feature_index2],"-",currX[feature_index3])

            Xnew.append(currX)
            ynew.append(curry)


        return [Xnew,ynew]