# Bounnoy Phanthavong (ID: 973081923)
# Homework 3
#
# This is a machine learning program that uses a Naive Bayes classifier
# based on Gaussians on sets of yeast data to predict which class they
# belong to.
# This program was built in Python 3.

from pathlib import Path
from math import e
import numpy as np
import csv
import pickle
import time
import sys

class NaiveBayes:
    def __init__(self, train, test):
        self.trainData = train
        self.testData = test

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
        f1 = np.genfromtxt(trainName + ".txt", delimiter=",")
        csv1 = open(trainName + ".pkl", 'wb')
        pickle.dump(f1, csv1)
        csv1.close()

    if not pklTest.exists():
        f2 = np.genfromtxt(testName + ".txt", delimiter=",")
        csv2 = open(testName + ".pkl", 'wb')
        pickle.dump(f2, csv2)
        csv2.close()

    file1 = open(trainName + ".pkl", "rb")
    train1 = pickle.load(file1)
    file1.close()

    file2 = open(testName + ".pkl", "rb")
    test2 = pickle.load(file2)
    file2.close()
