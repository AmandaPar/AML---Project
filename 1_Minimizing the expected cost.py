import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from costcla.metrics import cost_loss
from sklearn.calibration import CalibratedClassifierCV
from costcla.models import BayesMinimumRiskClassifier

data = pd.read_excel('cardio.xlsx', engine = 'openpyxl')

#--------------------------------data preprocessing----------------------------------------

#divide the 'age' variable to 10 classes, 1 for every 10 year period. 
for i in range(10):
    data.loc[(data['age'] >= (i*10*365)) & (data['age'] <= (i+1)*10*365) , 'age'] = i
    
#divide the 'ap_hi' (systolic blood pressure) variable to two classes, that of over and under 120.
data.loc[data['ap_hi'] < 120 , 'ap_hi'] = 0
data.loc[data['ap_hi'] >= 120 , 'ap_hi'] = 1
        
#divide the 'ap_lo' (diastolic blood pressure) variable to two classes, that of over and under 80.
data.loc[data['ap_lo'] < 80 , 'ap_lo'] = 0
data.loc[data['ap_lo'] >= 80 , 'ap_lo'] = 1
    
#drop unnecessary attributes 
X = data.drop(columns=['cardio','id','height'])
y = data['cardio']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
#fp, fn, tp, tn
# create an example-dependent cost-matrix required by costclas
fp = np.full((y_test.shape[0],1), 1)
fn = np.full((y_test.shape[0],1), 6)
tp = np.zeros((y_test.shape[0],1))
tn = np.zeros((y_test.shape[0],1))
cost_matrix = np.hstack((fp, fn, tp, tn))

#--------------------------------no cost minimization----------------------------------------

print("no cost minimization")
clf = LogisticRegression(solver="lbfgs")
model = clf.fit(X_train, y_train)
y_predicted = model.predict(X_test)
print("Recall Score: %.2f" % metrics.recall_score(y_test, y_predicted,average='macro'))
print("Precision Score: %.2f" % metrics.precision_score(y_test, y_predicted,average='macro'))
print("Accuracy Score: %.2f" % metrics.accuracy_score(y_test, y_predicted))
print("F1 Score: %.2f" % metrics.f1_score(y_test, y_predicted,average='macro'))
loss = cost_loss(y_test, y_predicted, cost_matrix)
print("%d\n" %loss)
print(confusion_matrix(y_test, y_predicted).T)

#---------------------------------no calibration---------------------------------------------

print("no calibration")
clf = KNeighborsClassifier(n_neighbors=200, weights='uniform', metric='minkowski', p=1)
model = clf.fit(X_train, y_train)
prob_test = model.predict_proba(X_test)
bmr = BayesMinimumRiskClassifier(calibration=False)
y_predicted = bmr.predict(prob_test, cost_matrix)
print("Recall Score: %.2f" % metrics.recall_score(y_test, y_predicted,average='macro'))
print("Precision Score: %.2f" % metrics.precision_score(y_test, y_predicted,average='macro'))
print("Accuracy Score: %.2f" % metrics.accuracy_score(y_test, y_predicted))
print("F1 Score: %.2f" % metrics.f1_score(y_test, y_predicted,average='macro'))
loss = cost_loss(y_test, y_predicted, cost_matrix)
print("%d\n" %loss)
print(confusion_matrix(y_test, y_predicted).T) 

#--------------------------------sigmoid calibration-----------------------------------------

print("\nsigmoid calibration")
clf = RandomForestClassifier(random_state=0, n_estimators=100)
cc = CalibratedClassifierCV(clf, method="sigmoid", cv=3)
model = cc.fit(X_train, y_train)
prob_test = model.predict_proba(X_test)
bmr = BayesMinimumRiskClassifier(calibration=False)
y_predicted = bmr.predict(prob_test, cost_matrix)
print("Recall Score: %.2f" % metrics.recall_score(y_test, y_predicted,average='macro'))
print("Precision Score: %.2f" % metrics.precision_score(y_test, y_predicted,average='macro'))
print("Accuracy Score: %.2f" % metrics.accuracy_score(y_test, y_predicted))
print("F1 Score: %.2f" % metrics.f1_score(y_test, y_predicted,average='macro'))
loss = cost_loss(y_test, y_predicted, cost_matrix)
print("%d\n" %loss)
print(confusion_matrix(y_test, y_predicted).T)

#--------------------------------isotonic calibration-----------------------------------------

print("\nisotonic calibration")
clf = RandomForestClassifier(random_state=0, n_estimators=100)
cc = CalibratedClassifierCV(clf, method="isotonic", cv=3)
model = cc.fit(X_train, y_train)
prob_test = model.predict_proba(X_test)
bmr = BayesMinimumRiskClassifier(calibration=False)
y_predicted = bmr.predict(prob_test, cost_matrix)
print("Recall Score: %.2f" % metrics.recall_score(y_test, y_predicted,average='macro'))
print("Precision Score: %.2f" % metrics.precision_score(y_test, y_predicted,average='macro'))
print("Accuracy Score: %.2f" % metrics.accuracy_score(y_test, y_predicted))
print("F1 Score: %.2f" % metrics.f1_score(y_test, y_predicted,average='macro'))
loss = cost_loss(y_test, y_predicted, cost_matrix)
print("%d\n" %loss)
print(confusion_matrix(y_test, y_predicted).T)