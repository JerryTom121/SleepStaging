"""
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def train(train_csv):
    train_mat = np.genfromtxt(train_csv, skip_header=0, dtype=str,delimiter=',')
    train_input = (train_mat[:,1:-1]).astype(float)
    train_output = (train_mat[:,-1]).astype(float)
    clf = LogisticRegression(class_weight='balanced')
    #clf = RandomForestClassifier(class_weight='balanced')
    clf.fit(train_input,train_output)
    truth = train_output
    preds = clf.predict(train_input)
    cmat = metrics.confusion_matrix(truth, preds)
    print "Shallow classification..."
    print "----------------------------------------"
    print "| EVAL: Artifact detection confusion matrix on training data:"
    print cmat
    print "----------------------------------------"
    print "| EVAL: Artifact detection evaluation on training data:"
    print "| Accuracy: " + format(metrics.accuracy_score(truth, preds, '.2f'))
    print "| Recall: " + format(cmat[0, 0]*1.0/(cmat[0, 0]+cmat[0, 1]), '.2f')
    print "| Precision: "+ format(cmat[0, 0]*1.0/(cmat[0, 0]+cmat[1, 0]), '.2f')
    print "----------------------------------------"
    
"""
# --------------------------------------------------------------------------- #
# ---------------------- Experimental setup --------------------------------- #
# --------------------------------------------------------------------------- #
aug  = '_aug'
CSV_FOLDER = '/home/djordje/Desktop/CSVdata/'
print CSV_FOLDER+'train_exp'+NEXP+aug+'.csv'
# --------------------------------------------------------------------------- #
# ---------------------- Read data ------------------------------------------ #
# --------------------------------------------------------------------------- #
train_mat = np.genfromtxt(CSV_FOLDER+'train_exp'+NEXP+aug+'.csv', skip_header=0, dtype=str, comments="4s",delimiter=',')
test_mat  = np.genfromtxt(CSV_FOLDER+'test_exp'+NEXP+'.csv', skip_header=0, dtype=str, comments="4s",delimiter=',')
# Training set
print "Load training data..."
train_data  = (train_mat[:,1:-1]).astype(float)
train_label = (train_mat[:,-1]).astype(float)
print "Shape is: "
print np.shape(train_data)
# Testing set
print "Load validation data..."
test_data  = (test_mat[:,1:-3]).astype(float)
test_label = (test_mat[:,-3]).astype(float)
print "Shape is: "
print np.shape(test_data)
print "----------------------------"
print "-- The augmentation type: " + aug 
print "----------------------------"
# --------------------------------------------------------------------------- #
# ---------------------- Choose classifier ---------------------------------- #
# --------------------------------------------------------------------------- #
# Random forest
#clf  = RandomForestClassifier();                  print "Random Forest..."
#clf  = GradientBoostingClassifier(n_estimators=1000)
clf  = LogisticRegression();                      print"Logistic Regression..."
#clf  = AdaBoostClassifier(n_estimators=200);       print"AdaBoost - 200"
#clf = SVC(C=5);										print"SVM..."
# --------------------------------------------------------------------------- #
# ---------------------- Test classification -------------------------------- #
# --------------------------------------------------------------------------- #
# Fit
clf.fit(train_data,train_label)
# Evaluate predictions
print "Evaluate on test set"
sleeplib.evaluate(clf.predict(test_data),test_label)
#print "Evaluate on train set"
#sleeplib.evaluate(clf.predict(train_data),train_label)
"""