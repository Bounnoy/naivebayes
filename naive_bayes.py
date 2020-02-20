# Bounnoy Phanthavong (ID: 973081923)
# Homework 3
#
# This is a machine learning program that uses a Naive Bayes classifier
# based on Gaussians on sets of yeast data to predict which class they
# belong to.
# This program was built in Python 3.

from pathlib import Path
import math
import numpy as np
import csv
import pickle
import time
import sys

class NaiveBayes:
    def __init__(self, train, test):
        self.trainData = train[train[:,(len(train[0])-1)].argsort()]
        self.testData = test

        print(self.trainData)

    def train(self):
        (classes, cnum) = np.unique(self.trainData[:,(len(train[0])-1)], return_counts=True)
        attribs = len(self.trainData[0])-1
        print(classes)
        print(cnum)
        print(attribs)

        # accuracy = np.zeros(len(classes)*attribs)         # Training accuracy.
        # accuracyTest = np.zeros(iterations)     # Test accuracy.

        for i in range(len(classes)):

            for x in range(attribs):
                a = 0 if (i - 1 < 0) else np.sum(cnum[:i-1])
                b = a + cnum[i]
                mean = np.sum(self.trainData[a:b,x]) / cnum[i]
                std = math.sqrt(np.sum( (self.trainData[a:b,x] - mean)**2 ) / cnum[i])
                std = 0.01 if (std == 0) else std
                print("Class", '%d'%classes[i] + ", mean =", '%.2f' % mean + ", std =", '%.2f' % std)

if __name__ == '__main__':

    if len(sys.argv) != 3:
        sys.exit("missing parameters: <train.txt> <test.txt>")

    trainName = str(sys.argv[1])[:-4]
    testName = str(sys.argv[2])[:-4]

    pklTrain = Path(trainName + ".pkl")
    pklTest = Path(testName + ".pkl")
    fileTrain = Path(trainName + ".txt")
    fileTest = Path(testName + ".txt")

    if not fileTrain.exists():
        sys.exit(trainName + ".txt not found")

    if not fileTest.exists():
        sys.exit(testName + ".txt not found")

    if not pklTrain.exists():
        f1 = np.genfromtxt(trainName + ".txt")
        csv1 = open(trainName + ".pkl", 'wb')
        pickle.dump(f1, csv1)
        csv1.close()

    if not pklTest.exists():
        f2 = np.genfromtxt(testName + ".txt")
        csv2 = open(testName + ".pkl", 'wb')
        pickle.dump(f2, csv2)
        csv2.close()

    file1 = open(trainName + ".pkl", "rb")
    train = pickle.load(file1)
    file1.close()

    file2 = open(testName + ".pkl", "rb")
    test = pickle.load(file2)
    file2.close()

    print(train)
    print("# of training rows/cols:", len(train), len(train[0]))
    print("# of test rows/cols:", len(test), len(test[0]))

    nb = NaiveBayes(train, test)
    nb.train()
