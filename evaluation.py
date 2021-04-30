
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
from skimage import io


mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

def analyze(prob_pred, y_true):

    y_pred = [np.argmax(prob) for prob in prob_pred]

    accuracy = accuracy_score(y_true, y_pred)
    global_precision = precision_score(y_true, y_pred, average = "micro")
    global_recall = recall_score(y_true, y_pred, average = "micro")
    global_f1 = f1_score(y_true, y_pred, average = "micro")
    weighted_roc_auc_ovr = roc_auc_score(y_true, prob_pred, multi_class="ovr", average="weighted")
    macro_roc_auc_ovr = roc_auc_score(y_true, prob_pred, multi_class="ovr", average="macro")
    weighted_roc_auc_ovo = roc_auc_score(y_true, prob_pred, multi_class="ovo", average="weighted")
    macro_roc_auc_ovo = roc_auc_score(y_true, prob_pred, multi_class="ovo", average="macro")

    print(classification_report(y_true, y_pred))
    print("global recall: {}".format(global_recall))
    print("global precision: {}".format(global_precision))
    print("global f1: {}".format(global_f1))
    print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} (weighted by prevalence)"
      .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
    print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} (weighted by prevalence)"
      .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))

    conf_mx = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(conf_mx).plot()
    plt.show()

    show_command = input("See the wrong answers? (y/n)")
    i = 0
    while i < len(y_pred) and show_command == 'y':
        if y_pred[i] != y_true[i]:
                print('predicted:', y_pred[i], 'target:', y_true[i])
                io.imshow(np.squeeze(X_test[i]))
                io.show()
                show_command = input("See the wrong answers? (y/n)")
        i+=1

    return conf_mx

def first_image_predict(model, X_test):
    #making a test set that only includes one element for each frame sequence
    shape = X_test.shape
    X_first_test = np.zeros((shape[0], shape[2], shape[3], shape[4]))

    #the test set only includes the first image of each frame sequence
    for i, sequence in enumerate(X_test):
        X_first_test[i] = sequence[0]

    prob_pred = model.predict(X_first_test)

    return prob_pred

def probability_predict(model, X_test):
    #making a test set that only includes one element for each frame sequence
    shape = X_test.shape
    X_prob_test = np.zeros((shape[0], shape[2], shape[3], shape[4]))

    #making the test set based on the probability method
    for i, sequence in enumerate(X_test):
        #class probabilities for each frame in the sequence
        probabilities = model.predict(sequence)
        #maximum probability for each frame in the sequence
        predictions = probabilities.argmax(axis=-1)
        #the total probability of each class over all sequences
        sum_prob = np.sum(probabilities, axis=0)
        #the class with the highest total probability
        prob_prediction = sum_prob.argmax(axis=-1)

        for j, prediction in enumerate(predictions):
            if prediction == prob_prediction:
                X_prob_test[i] = sequence[j] #inserting a desired image on prob_test
                break
    prob_pred = model.predict(X_prob_test)
    return prob_pred

def vote_predict(model, X_test):
    #making a test set that only includes one element for each frame sequence
    shape = X_test.shape
    X_vote_test = np.zeros((shape[0], shape[2], shape[3], shape[4]))

    #making the test set based on the voting method
    for i, sequence in enumerate(X_test):
        #class probabilities for each frame in the sequence
        probabilities = model.predict(sequence)
        #maximum probability for each frame in the sequence
        predictions = probabilities.argmax(axis=-1)
        #the most common prediction of all frames
        common_key = Counter(predictions).most_common(1)[0][0]

        for j, prediction in enumerate(predictions):
            if prediction == common_key:
                X_vote_test[i] = sequence[j] #inserting a desired image on vote_test
                break
    prob_pred = model.predict(X_vote_test)
    return prob_pred


#the below functions were initially written for getting the overall performance after cross-validation, but I found a better way
def sum_confusion_matrix(matrices):
    num_matrices = matrices.shape[0]
    num_labels = matrices.shape[1]
    sum_conf = np.zeros((num_labels, num_labels))

    for i in range(num_matrices):
        for j in range(num_labels):
            for k in range(n):
                sum_conf[j][k] += matrices[i][j][k]
    print(sum_conf)
    return sum_conf

def get_precisions(conf_mx):
    num_labels = conf_mx.shape[0]
    precisions = np.zeros(num_labels)
    for i in range(num_labels):
        true_pos = conf_mx[i][i]
        total_pos = 0
        for j in range(num_labels):
            total_pos += conf_mx[j][i]
        precisions[i] = true_pos/total_pos
    return precisions

def get_recalls(conf_mx):
    num_labels = conf_mx.shape[0]
    recalls = np.zeros(num_labels)
    for i in range(num_labels):
        true_pos = conf_mx[i][i]
        total_class = 0
        for j in range(num_labels):
            total_class += conf_mx[i][j]
        recalls[i] = true_pos/total_class
    return recalls

def get_f1_scores(conf_mx):
    num_labels = conf_mx.shape[0]
    f1_scores = np.zeros(num_labels)
    precisions = get_precisions(conf_mx)
    recalls = get_recalls(conf_mx)

    for i in range(num_labels):
        f1_scores[i] = (2 * precisions[i] * recalls[i])/(precisions[i] + recalls[i])

    return f1_scores

def get_accuracy(conf_mx):
    num_labels = conf_mx.shape[0]
    true_pos = 0
    total_preds = 0
    for i in range(num_labels):
        for j in range(num_labels):
            total_preds += conf_mx[i][j]
            if i == j:
                true_pos += conf_mx[i][j]
    accuracy = true_pos / total_preds
    return accuracy
