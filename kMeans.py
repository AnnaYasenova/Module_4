import math
import sys
from random import shuffle, uniform, choice
from matplotlib import pyplot
import numpy as np

def loadData(fileName):
    # reading data from file
    f = open(fileName, 'r')
    lines = f.read().splitlines()
    f.close()

    items = []

    for i in range(1, len(lines)):
        line = lines[i].split(',')
        itemDescr = []
        for j in range(len(line) - 1):
            v = float(line[j])
            itemDescr.append(v)
        items.append(itemDescr)

    shuffle(items)
    return items


# supporting function
def findMinMaxCol(items):
    n = len(items[0])
    minVal = [sys.maxsize for i in range(n)]
    maxVal = [-sys.maxsize - 1 for i in range(n)]

    for item in items:
        for el in range(len(item)):
            if (item[el] < minVal[el]):
                minVal[el] = item[el]

            if (item[el] > maxVal[el]):
                maxVal[el] = item[el]
    return minVal, maxVal


# calculating euclidean distance
def euclidDist(x, y):
    S = 0
    for i in range(len(x)):
        S += math.pow(x[i] - y[i], 2)
    return math.sqrt(S)


# initializing means to random number from [min;max] of each feature
def initMeans(items, k, minCol, maxCol):
    num = len(items[0])
    means = [[0 for i in range(num)] for j in range(k)]

    for mean in means:
        for i in range(len(mean)):
            # uniform - Draw samples from a uniform distribution. [docs] :)
            mean[i] = uniform(minCol[i] + 1, maxCol[i] - 1)
    return means

# calculating
def calcMean(n, mean, item):
    for i in range(len(mean)):
        m = mean[i]
        m = (m * (n - 1) + item[i]) / float(n)
        mean[i] = round(m, 3)
    return mean


def findClusters(means, items):
    result = [[] for i in range(len(means))]
    # Classifying each item into a cluster and adding item to cluster
    for item in items:
        index = Classify(means, item)
        result[index].append(item)
    return result


# MAGIC FUNCTION. find mean with minimum distance and clasify item
def Classify(means, item):
    minimum = sys.maxsize
    index = -1
    for i in range(len(means)):
        dist = euclidDist(item, means[i])
        if (dist < minimum):
            minimum = dist
            index = i
    return index


def calcMeans(k, items, maxIterations=100000):
    minCol, maxCol = findMinMaxCol(items)
    means = initMeans(items, k, minCol, maxCol)
    print(means)
    # clusterCounters - number of item in class
    clusterCounters = [0 for i in range(len(means))]
    # element-cluster
    includedIn = [0 for i in range(len(items))]

    for e in range(maxIterations):
        flag = True # to show changes
        for i in range(len(items)):
            item = items[i]
            index = Classify(means, item)
            clusterCounters[index] += 1
            means[index] = calcMean(clusterCounters[index], means[index], item)
            if (index != includedIn[i]):
                flag = False
            includedIn[i] = index
        if (flag):
            break
    return means


def plotting(clusters):
    n = len(clusters)
    # save two dimensions
    X = [[] for i in range(n)]

    for i in range(n):
        cluster = clusters[i]
        for item in cluster:
            X[i].append(item)

    colors = ['r', 'y', 'b', 'g']
    counter = 1
    for x in X:
        c = choice(colors)
        #remove to avoid duplicates
        colors.remove(c)
        Xx = []
        Xy = []
        for item in x:
            Xx.append(item[0])
            Xy.append(item[1])
        pyplot.plot(Xx, Xy, 'o', color = c, label=str(counter)+' - '+c)
        counter += 1
    pyplot.legend()
    pyplot.show()

def main():
    items = loadData('iris.csv')

    k = 3
    means = calcMeans(k, items)
    clusters = findClusters(means, items)
    plotting(clusters)
    print('Means: ', means)
    print('Cluster: ', clusters)

    # check
    newItem = [5.4, 3.7, 1.5, 0.2]
    print('New item cluster: ', Classify(means, newItem))

if __name__ == "__main__":
    main()
