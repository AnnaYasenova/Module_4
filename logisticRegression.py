from collections import Counter
import pandas as pd
import numpy as np

# remove quotes
def strip_quotations_newline(text):
    text = text.rstrip()
    if text[0] == '"':
        text = text[1:]
    if text[-1] == '"':
        text = text[:-1]
    return text

# add spaces
def expand_around_chars(text, characters):
    for char in characters:
        text = text.replace(char, " " + char + " ")
    return text

def split_text(text):
    text = strip_quotations_newline(text)
    text = expand_around_chars(text, '".,()[]{}:;')
    splitted_text = text.split(" ")
    cleaned_text = [x for x in splitted_text if len(x) > 1]
    text_lowercase = [x.lower() for x in cleaned_text]
    return text_lowercase

def pow10(x):
    num = 1;
    while ((num * 10) < x):
        num *= 10.0;
    return num

def normalize_col(col, method):
    colMean = np.mean(col)
    if method == 'pow10':
        return col / pow10(np.max(col))
    else:
        return col - colMean

def normalize_matrix(X):
    nRow, nCol = np.shape(X)
    X_norm = np.zeros(shape=(nRow, nCol))
    X_norm[:, 0] = X[:, 0]
    for i in range(1, nCol):
        X_norm[:, i] = normalize_col(X[:, i], 'mean')
    return X_norm

# load dataset
def iris():
    df = pd.read_csv('iris.csv', header = None)
    # 70% train, 30% test
    df_train = df.sample(frac = 0.7)
    df_test = df.loc[~df.index.isin(df_train.index)]
    X_train = df_train.values[:, 0:4].astype(float)
    Y_train = df_train.values[:, 4]
    X_test = df_test.values[:, 0:4].astype(float)
    Y_test = df_test.values[:, 4]
    return X_train, Y_train, X_test, Y_test

def true_positives(determined_Y, real_Y, label):
    true_positives = 0
    for ii in range(0, len(determined_Y)):
        if determined_Y[ii] == label and real_Y[ii] == label:
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


class LogisticRegression():

    def __init__(self, learning_rate=0.7, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.theta = []
        self.examplesNum = 0
        self.featuresNum = 0
        self.X = None
        self.Y = None

    def add_bias_col(self, X):
        bias_col = np.ones((X.shape[0], 1))
        return np.concatenate([bias_col, X], axis=1)

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-1.0 * np.dot(X, self.theta)))

    def cost_function(self):
        predicted_Y = self.sigmoid(self.X)
        cost = (-1.0 / self.examplesNum) * np.sum(
            self.Y * np.log(predicted_Y) + (1 - self.Y) * (np.log(1 - predicted_Y)))
        return cost

    def gradient(self):
        predicted_Y_values = self.sigmoid(self.X)
        grad = (-1.0 / self.examplesNum) * np.dot((self.Y - predicted_Y_values), self.X)
        return grad

    def gradient_descent(self):
        for iter in range(1, self.max_iter):
            cost = self.cost_function()
            delta = self.gradient()
            self.theta = self.theta - self.learning_rate * delta
            print("iteration %s : cost %s " % (iter, cost))

    def train(self, X, Y):
        self.X = self.add_bias_col(X)
        self.Y = Y
        self.examplesNum, self.featuresNum = np.shape(X)
        self.theta = np.ones(self.featuresNum + 1)
        self.gradient_descent()

    def classify(self, X):
        X = self.add_bias_col(X)
        predicted_Y = self.sigmoid(X)
        predicted_Y_binary = np.round(predicted_Y)
        return predicted_Y_binary

def main():
    arr = {1: {'Iris-setosa': 1, 'Iris-versicolor': 0, 'Iris-virginica': 0},
                2: {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 0},
                3: {'Iris-setosa': 0, 'Iris-versicolor': 0, 'Iris-virginica': 1}
                }

    X_train, y_train, X_test, y_test = iris()

    Y_train = np.array([arr[3][x] for x in y_train])
    Y_test = np.array([arr[3][x] for x in y_test])

    print("training Logistic Regression")
    lr = LogisticRegression()
    lr.train(X_train, Y_train)
    print("trained")
    predicted_Y_test = lr.classify(X_test)

    f1 = f1_score(predicted_Y_test, Y_test, 1)
    print("F1-score on the test-set for class %s is: %s" % (1, f1))

if __name__ == '__main__':
    main()