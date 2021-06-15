from skmultilearn.problem_transform import BinaryRelevance

def binary_run(classifier, train_test_set):
    X_train, X_test, y_train, y_test = train_test_set

    binary_relevance = BinaryRelevance(classifier)

    binary_relevance.fit(X_train, y_train)

    y_pred = binary_relevance.predict(X_test)

    return y_test, y_pred