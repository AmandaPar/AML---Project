from skmultilearn.problem_transform import ClassifierChain


def classifier_run(classifier, train_test_set):

    X_train, X_test, y_train, y_test = train_test_set

    # initialize classifier chains multi-label classifier
    # with a gaussian naive bayes base classifier
    chain = ClassifierChain(classifier, order='random', random_state=0)
    # train
    chain.fit(X_train, y_train)

    # predict
    y_pred = chain.predict(X_test)

    return y_test, y_pred
