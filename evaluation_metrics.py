from sklearn import metrics

"""
The following function, presents all the metrics we used to evaluate the model
@ param : actual
@ param : predicted 
"""

def evaluate_metrics_model(actual, predicted, average="micro", print_results=False):

    recall_score = metrics.recall_score(actual, predicted, average=average)
    f1_score = metrics.f1_score(actual, predicted, average=average)
    accuracy_score = metrics.accuracy_score(actual, predicted)
    hamming_loss = metrics.hamming_loss(actual, predicted)
    precision_score = metrics.precision_score(actual, predicted, average=average)
    classfication_report = metrics.classification_report(actual, predicted)

    if print_results:
        print(f"Accuracy score: {accuracy_score:.2f}")
        print(f"Precision score: {precision_score:.2f} with average parameter: {average}")
        print(f"Recall score: {recall_score:.2f} with average parameter: {average}")
        print(f"F1 score: {f1_score:.2f} with average parameter: {average}")
        print(f"Hamming loss: {hamming_loss:.2f}")
        print(f"Classification report:\n{classfication_report}")

    return accuracy_score, precision_score, recall_score, f1_score, hamming_loss, classfication_report

"""
The following function, calculates the metrics for every label will be predict
@ param : actual
@ param : predicted 
"""
def evaluate_metrics_per_label(actual, predicted, print_results=False):
    precision_score_per_label = []
    recall_score_per_label = []

    accuracy_score_per_label = []
    f1_score_per_label = []

    for label in range(actual.shape[1]):
        actual_label = actual.values[:, label]
        predicted_label = predicted[:, label]

        recall_score = metrics.recall_score(actual_label, predicted_label)
        precision_score = metrics.precision_score(actual_label, predicted_label)
        f1_score = metrics.f1_score(actual_label, predicted_label)
        accuracy_score = metrics.accuracy_score(actual_label, predicted_label)

        accuracy_score_per_label.append(accuracy_score)
        precision_score_per_label.append(precision_score)
        recall_score_per_label.append(recall_score)
        f1_score_per_label.append(f1_score) 

        if print_results:
            print(f"Accuracy score: {accuracy_score:.2f}")
            print(f"Precision score: {precision_score:.2f}")
            print(f"Recall score: {recall_score:.2f}")
            print(f"F1 score: {f1_score:.2f}")

    return accuracy_score_per_label, precision_score_per_label, recall_score_per_label, f1_score_per_label


def run():
    pass

if __name__ == "__main__":
    run()