from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import confusion_matrix
from costcla.metrics import cost_loss
from sklearn import metrics
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np

data = pd.read_excel('cardio.xlsx', engine='openpyxl')

# --------------------------------data preprocessing----------------------------------------

# divide the 'age' variable to 10 classes, 1 for every 10 year period.
for i in range(10):
    data.loc[(data['age'] >= (i * 10 * 365)) & (data['age'] <= (i + 1) * 10 * 365), 'age'] = i

# divide the 'ap_hi' (systolic blood pressure) variable to two classes, that of over and under 120.
data.loc[data['ap_hi'] < 120, 'ap_hi'] = 0
data.loc[data['ap_hi'] >= 120, 'ap_hi'] = 1

# divide the 'ap_lo' (diastolic blood pressure) variable to two classes, that of over and under 80.
data.loc[data['ap_lo'] < 80, 'ap_lo'] = 0
data.loc[data['ap_lo'] >= 80, 'ap_lo'] = 1

# drop unnecessary attributes
X = data.drop(columns=['cardio', 'id', 'height'])
y = data['cardio']

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=42)
# fp, fn, tp, tn
# create an example-dependent cost-matrix required by costclas
fp = np.full((y_test.shape[0], 1), 1)
fn = np.full((y_test.shape[0], 1), 6)
tp = np.zeros((y_test.shape[0], 1))
tn = np.zeros((y_test.shape[0], 1))
cost_matrix = np.hstack((fp, fn, tp, tn))

# create a classic cost-matrix
cost_m = [[0, 6], [1, 0]]

###'linear SVM','MultinomialNB',
### SVC(kernel='linear', C=1), MultinomialNB(alpha=0.05)

names = ['LogisticRegression', 'random forest', 'KNeighborsClassifier', 'MLPClassifier']
classifiers = [LogisticRegression(solver="lbfgs"),
               RandomForestClassifier(n_estimators=100, random_state=0),
               KNeighborsClassifier(n_neighbors=200, weights='uniform', metric='minkowski', p=1),
               MLPClassifier(hidden_layer_sizes=(50, 50, 50), activation='tanh', solver='lbfgs', max_iter=100,
                             tol=0.00001)]

for name, clf in zip(names, classifiers):
    print(name)
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    print("Recall Score: %.2f" % metrics.recall_score(y_test, y_predicted, average='macro'))
    print("Precision Score: %.2f" % metrics.precision_score(y_test, y_predicted, average='macro'))
    print("Accuracy Score: %.2f" % metrics.accuracy_score(y_test, y_predicted))
    print("F1 Score: %.2f" % metrics.f1_score(y_test, y_predicted, average='macro'))

    conf_m = confusion_matrix(y_test, y_predicted).T
    print(conf_m)
    print(np.sum(conf_m * cost_m))
    loss = cost_loss(y_test, y_predicted, cost_matrix)
    print("%d\n" % loss)