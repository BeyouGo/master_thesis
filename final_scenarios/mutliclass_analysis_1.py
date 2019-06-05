##################################################################################################
#########################     Scenario 3: multi_class analysis         ###########################
##################################################################################################
from sklearn.tree import tree
import time
from numpy import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from EncryptedTrafficClassification.Dataset import Dataset


def get_model():
    model = tree.DecisionTreeClassifier(max_depth=30)
    return model

def get_best_features():
    features = []
    features.append('Pkt Len Max')
    features.append('Flow IAT Max')
    # features.append('Src Port')
    # features.append('Dst Port')
    return features


dataset = Dataset("..\\flows\\")

class_names = ['aim', 'email', 'facebook', 'ftps', 'hang', 'icq', 'netflix', 'scp', 'sftp', 'skype','spotify', 'vimeo', 'voip', 'youtube']
################ 0 ##### 1 ######## 2 ####### 3 ##### 4 #### 5 ####### 6 ##### 7 ##### 8 ##### 9 ###### 10 ##### 11 ##### 12 ###### 13 ###

features = get_best_features()
# Xall, Yall = dataset.get_balanced_data(range(0,14), features)
Xall, Yall = dataset.get_balanced_data(range(0,14), features)

### Xall and Yall contain the flows of all the files
test_size = 0.3
X_train, X_test, Y_train, Y_test = train_test_split(Xall, Yall, test_size=test_size, random_state=random.randint(1000),shuffle=True)

# model = tree.DecisionTreeClassifier(max_depth=30,criterion='gini',random_state=0)
model = RandomForestClassifier(n_estimators=200, random_state=0,criterion='entropy')#,criterion="entropy",
# model = GradientBoostingClassifier()
training_start = time.time()
model.fit(X_train, Y_train)
training_end = time.time()

### Compute score using test set
result = model.score(X_test, Y_test)
# print(result)
testing_start = time.time()
pred_lab_test = model.predict(X_test)
testing_end = time.time()
test_accuracy = np.mean(pred_lab_test.ravel() == Y_test.ravel())
print(pred_lab_test.ravel()[:20])
print(Y_test.ravel()[:20])

###############################################################

class_names = ['aim', 'email', 'facebook', 'ftps', 'hang', 'icq', 'netflix', 'scp', 'sftp', 'skype',
               'spotify', 'vimeo', 'voip', 'youtube']

print("training set size:", np.bincount(Y_train))
print("Accuracy :",test_accuracy)
print("Training time :", training_end - training_start)
print("Testing time :", testing_end-testing_start)

#
# confmat = confusion_matrix(Y_test.ravel(),pred_lab_test.ravel())
#
# print(confmat)
# np.savetxt('confmat.csv',
#            confmat,
#            delimiter=',',
#            fmt='%3i')