from sklearn.multiclass import OneVsRestClassifier

def oneRest_run(classifier, train_test_set):
    X_train, X_test, y_train, y_test = train_test_set

    # init model and fit to train data
    model = OneVsRestClassifier(classifier)
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)
    print('\n--------OneVsRest with {:}'.format(model.estimator))

    return y_test, y_pred