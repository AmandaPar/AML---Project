import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from costcla.metrics import cost_loss
from sklearn import metrics
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve

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

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.01, random_state=42)

# Create classifiers
lr = LogisticRegression()
gnb = GaussianNB()
svc = LinearSVC(C=1.0)
rfc = RandomForestClassifier(n_estimators=100, random_state=0)

#kn = KNeighborsClassifier(n_neighbors=200, weights='uniform', metric='minkowski', p=1), 
mlp = MLPClassifier(hidden_layer_sizes=(50, 50, 50), activation='tanh', solver='lbfgs', max_iter=100, tol=0.00001)


# #############################################################################
# Plot calibration plots

plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
for clf, name in [(lr, 'Logistic'),
                  (gnb, 'Naive Bayes'),
                  (rfc, 'Random Forest'),
                  (mlp, 'MLPClassifier'),
                  ]:
    clf.fit(X_train, y_train)
    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(X_test)[:, 1]
    else:  # use decision function
        prob_pos = clf.decision_function(X_test)
        prob_pos = \
            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(y_test, prob_pos, n_bins=10)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s" % (name, ))

    ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
             histtype="step", lw=2)

ax1.set_ylabel("Fraction of positives")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.set_title('Calibration plots  (reliability curve)')

ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")
ax2.legend(loc="upper center", ncol=2)

plt.tight_layout()
plt.show()