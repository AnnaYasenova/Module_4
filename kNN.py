import csv
import random
import math
import operator

def loadData(filename, split):
    trainingSet = []
    testSet = []
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
        return trainingSet, testSet

# calculating euclidean distance
def euclidDist(point1, point2, length):
    dist = 0
    for x in range(length):
        dist += pow((point1[x] - point2[x]), 2)
    return math.sqrt(dist)

# find k distanses
def getNeighborsDist(trSet, testPoint, k):
    distances = []
    length = len(testPoint) - 1
    # find all distanses
    for x in range(len(trSet)):
        dist = euclidDist(testPoint, trSet[x], length)
        distances.append((trSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResult(neighbors):
    classNum = {}
    for x in range(len(neighbors)):
        item = neighbors[x][-1]
        if item in classNum:
            classNum[item] += 1
        else:
            classNum[item] = 1
    # REVERSE sort for find
    sortedClasses = sorted(classNum.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClasses[0][0]

# calculate accuracy
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def main():
    # preparing data
    # change split, if you need
    split = 0.7
    trainingSet, testSet = loadData('iris.csv', split)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
    # generating predictions
    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighborsDist(trainingSet, testSet[x], k)
        result = getResult(neighbors)
        predictions.append(result)
        print('result: ' + repr(result) + ', real (in test set):' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')

if __name__ == '__main__':
    main()