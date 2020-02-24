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
import random

class NaiveBayes:
    def __init__(self, train, test):
        self.trainData = train[train[:,(len(train[0])-1)].argsort()]    # Sort data by class.
        self.testData = test

    # This class computes the gaussian for each test data and returns the predicted class and probability.
    def gaussian(self, row, list, pc):
        pred = 0    # Highest prediction
        prob = 0.0  # Gaussian of predicted class.
        acc = 0.0   # Accuracy of our prediction.
        ties = 0    # Count of how many ties for the predicted class.
        correct = 0 # Flag that marks our prediction as correct or not.
        tgaus = 0   # Total gaussian after calculating for all classes.

        # Loop through classes.
        for i in range(len(list)):
            gaus = 0.0                      # Current gaussian.
            target = int(row[len(row)-1])   # Our target output.

            # Loop through attrributes.
            for j in range(len(list[0])):
                mean = list[i][j][0]        # Extract the mean from our training list.
                std = list[i][j][1]         # Extract the standard deviation from our training list.

                x = (row[j] - mean) / std                                   # Subcalculation for exponent.
                calc = math.exp(-x*x/2.0) / (math.sqrt(2.0*math.pi) * std)  # Calculate full gaussian.
                gaus += math.log(calc) if calc > 0 else 0.0                # Log results to prevent underflow.

            gaus += math.log(pc[i]) # Add probability(class).
            gaus = math.exp(gaus)   # Raise the results back so they're non-negative.
            tgaus += gaus           # Add to our total gaussian counter.

            # Check if new gaussian is about the same as the highest predicted gaussian.
            if math.isclose(prob, gaus):
                ties += 1
                pred = random.choice([pred, list[i][0][2]]) # Randomly pick a prediction between the ties.
            else:
                prob = max(prob, gaus)
                # If the new gaussian is highest, make it the new predicted class.
                # Else, keep old prediction.
                pred = list[i][0][2] if math.isclose(prob, gaus) else pred

            if target == pred:
                correct = 1
            else:
                correct = 0

        # If no ties and correct, 1.
        # If no ties and incorrect, then 0.
        # If ties and correct, 1/n.
        # If ties and incorrect, 0
        if ties == 0 and correct > 0:
            acc = 1
        elif ties > 0 and correct > 0:
            acc = 1/ties
        else:
            acc = 0

        return pred, prob/tgaus, acc

    def train(self):
        # Grabs a tuple containing the class and how many rows of each class we have.
        (classes, cnum) = np.unique(self.trainData[:,(len(self.trainData[0])-1)], return_counts=True)

        attribs = len(self.trainData[0])-1  # Number of attributes in data set.

        # Probability of each class in the dataset.
        pc = np.zeros(len(classes))
        for a in range(len(pc)):
            pc[a] = cnum[a] / len(self.trainData)

        # List of mean, standard deviation, and class.
        msList = np.zeros((len(classes), attribs, 3))

        # Loop through each class in the training data.
        for i in range(len(classes)):

            # Loop through each attribute.
            for x in range(attribs):

                # Offset from a to b. Used for slicing magic.
                a = 0 if (i - 1 < 0) else np.sum(cnum[:i])
                b = a + cnum[i]

                # Calculate our mean and standard deviation.
                mean = np.sum(self.trainData[a:b,x], dtype=float) / cnum[i]
                std = math.sqrt(np.sum( (self.trainData[a:b,x] - mean)**2, dtype=float ) / cnum[i])
                std = 0.01 if (std < 0.01) else std

                # Add the mean, standard deviation, and class to msList.
                msList[i][x][0] = mean
                msList[i][x][1] = std
                msList[i][x][2] = classes[i]

                print("Class", '%d' % classes[i] + ", attribute", '%d' % (x+1) + ", mean =", '%.2f' % mean + ", std =", '%.2f' % std)

        accuracy = 0

        # Rows in Test Data
        for j in range(len(self.testData)):
            pred, prob, acc = self.gaussian(self.testData[j], msList, pc)
            truth = self.testData[j][len(self.testData[0])-1]
            accuracy += acc
            print("ID=" + '%5d' % (j+1) + ", predicted=" + '%3d' % pred + ", probability =", '%.4f' % prob + ", true=" + '%3d' % truth + ", accuracy=" + '%4.2f' % acc)

        print("classification accuracy=" + '%6.4f' % (accuracy/len(self.testData)*100))

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

    nb = NaiveBayes(train, test)
    nb.train()
