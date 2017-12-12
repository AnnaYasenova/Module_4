from collections import Counter, defaultdict
import numpy as np
import pandas as pd


def load_data():
    datafile = 'adult_data.csv'
    file_test = 'adult_test.csv'
    df = pd.read_csv(datafile, header=None)
    Y_train = df[14].values
    del df[14]
    del df[2]
    X_train = df.values

    df_test = pd.read_csv(file_test, header=None)
    Y_test = df_test[14].values
    del df_test[14]
    del df_test[2]
    X_test = df_test.values
    return X_train, Y_train, X_test, Y_test

""" ERROR MATRIX

          | y = 1  |  y = -1
----------+--------+-----------
a(x) = 1  |  TP    |    TN
----------+--------+-----------
a(x) = -1 |  FN    |    FP
"""
def true_positives(determined_Y, real_Y, label):
    true_positives = 0
    for i in range(0, len(determined_Y)):
        if determined_Y[i] == label and real_Y[i] == label:
            true_positives += 1
    return true_positives


def all_positives(determined_Y, label):
    return Counter(determined_Y)[label]


def false_negatives(determined_Y, real_Y, label):
    false_negatives = 0
    for i in range(0, len(determined_Y)):
        if determined_Y[i] != label and real_Y[i] == label:
            false_negatives += 1
    return false_negatives


def precision(determined_Y, real_Y, label):
    if float(all_positives(determined_Y, label)) == 0:
        return 0
    return true_positives(determined_Y, real_Y, label) / float(all_positives(determined_Y, label))


def recall(determined_Y, real_Y, label):
    denominator = float((true_positives(determined_Y, real_Y, label) + false_negatives(determined_Y, real_Y, label)))
    if denominator == 0:
        return 0
    return true_positives(determined_Y, real_Y, label) / denominator


def f1_score(determined_Y, real_Y, label=1):
    p = precision(determined_Y, real_Y, label)
    r = recall(determined_Y, real_Y, label)
    if p + r == 0:
        return 0
    f1 = 2 * (p * r) / (p + r)
    return f1


class NaiveBaseClass:
    def calculate_relative_occurences(self, lst):
        examplesNum = len(lst)
        ro_dict = dict(Counter(lst))
        for key in ro_dict.keys():
            ro_dict[key] = ro_dict[key] / float(examplesNum)
        return ro_dict

    def get_max_value_key(self, d1):
        values = list(d1.values())
        keys = list(d1.keys())
        max_value_index = values.index(max(values))
        max_key = keys[max_value_index]
        return max_key

    def initialize_nb_dict(self):
        self.nb_dict = {}
        for label in self.labels:
            self.nb_dict[label] = defaultdict(list)


class NaiveBayes(NaiveBaseClass):
    def train(self, X, Y):
        self.labels = np.unique(Y)
        rowNum, colNum = np.shape(X)
        self.initialize_nb_dict()
        self.class_probabilities = self.calculate_relative_occurences(Y)
        for label in self.labels:
            row_indices = np.where(Y == label)[0]
            X_ = X[row_indices, :]
            rowNum_, colNum_ = np.shape(X_)
            for i in range(0, colNum_):
                self.nb_dict[label][i] += list(X_[:, i])
        for label in self.labels:
            for j in range(0, colNum):
                self.nb_dict[label][j] = self.calculate_relative_occurences(self.nb_dict[label][j])


    def classify_single_elem(self, X_elem):
        Y_dict = {}
        for label in self.labels:
            class_probability = self.class_probabilities[label]
            for i in range(0, len(X_elem)):
                relative_feature_values = self.nb_dict[label][i]
                if X_elem[i] in relative_feature_values.keys():
                    class_probability *= relative_feature_values[X_elem[i]]
                else:
                    class_probability *= 0
            Y_dict[label] = class_probability
        return self.get_max_value_key(Y_dict)

    def classify(self, X):
        self.predicted_Y_values = []
        rowNum, colNum = np.shape(X)
        for i in range(0, rowNum):
            X_elem = X[i, :]
            prediction = self.classify_single_elem(X_elem)
            self.predicted_Y_values.append(prediction)
        return self.predicted_Y_values


def main():
    X_train, Y_train, X_test, Y_test = load_data()
    print('Training')
    print('...............')
    nbc = NaiveBayes()
    nbc.train(X_train, Y_train)
    print('Trained!')
    predicted_Y = nbc.classify(X_test)
    y_labels = np.unique(Y_test)
    for y_label in y_labels:
        f1 = f1_score(predicted_Y, Y_test, y_label)
        print('F1-score on the test-set for class %s is: %s' % (y_label, f1))

if __name__ == '__main__':
    main()