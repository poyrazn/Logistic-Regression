#
# assignment3
# ML_Assignment3
#
# Created by Nehir Poyraz 21.03.2018
# Copyright © 2018 Nehir Poyraz. All rights reserved.


import os.path
import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm


class classifier:


    def __init__(self, labels):
        self.mt_covariance = []
        self.mt_mean = []
        self.labels = labels


    def train(self, data):
        """ trains the classifier with given data, constructs mean vectors """
        for i in range(len(data)):
            self.mt_covariance.append(covariance(data[i]))
            self.mt_mean.append(meanvec(data[i]))

    def test(self, data):
        """
        tests the classifier on test data, returns the number of errors

        :param data:    data(list)
        :return:        error (int)
        """
        error = 0
        decision = [["" for datum in data[0]] for j in range(len(data))]
        for i in range(len(data)):
            for j in range(len(data[i])):
                disc = []
                for k in range(len(data)):
                    disc.append(discriminant(data[i][j], i, self.mt_covariance[i], self.mt_mean[k]))
                maximum = max(disc)
                decision[i][j] = self.labels[disc.index(maximum)]
                if not disc.index(maximum) == i:
                    error += 1
        # print(decision)
        return error

    def printclf(self):
        """
        prompts console output:     Covariance matrices and mean vectors for each class
        """

        for i in range(len(self.mt_covariance)):
            cov = self.mt_covariance[i]
            mean = self.mt_mean[i]
            print("\nCovariance matrix", i+1)
            for j in range(len(cov)):
                for k in range(len(cov)):
                    print(round(cov[j][k], 1), end=" ")
                print()
            print("\nMean", i+1)
            for j in range(len(mean)):
                print("%.f" % mean[j])

    def plotclf(self, totalsize, name):
        """
        plots data, samples and decision  boundaries
        :param totalsize: number of samples in the dataset
        :return:
        """
        samples = []
        samplesize = int(totalsize/len(self.labels))

        for i in range(len(self.mt_mean)):
            mu_vec = np.array(self.mt_mean[i])
            cov_mat = np.array(self.mt_covariance[i])
            samples.append(np.random.multivariate_normal(mu_vec, cov_mat, samplesize))
        colors = ['blue', 'orange', 'green', 'red', 'm', 'k']
        col_ix = 0
        for sample in samples:
            plt.scatter(sample[:, 0], sample[:, 1], cmap=plt.cm.coolwarm, s=1, marker='.', c=colors[col_ix % len(colors)])
            col_ix += 1

        X = np.concatenate(samples, axis=0)
        Y = []
        for i in range(len(samples)):
            for j in range(samplesize):
                Y.append(i)

        C = 1.0  # SVM regularization parameter
        # clf = svm.SVC(kernel='rbf', gamma=.002, C=C)
        clf = svm.SVC(kernel='poly', degree=2, C=C)
        clf.fit(X, Y)
        X0, X1 = X[:, 0], X[:, 1]
        xx, yy = make_meshgrid(X0, X1)
        plot_contours(plt, clf, xx, yy, alpha=0.2)
        # fig=plt.figure(name)
        plt.title(name)
        plt.show()


def make_meshgrid(x, y, h=.1):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 10, x.max() + 10
    y_min, y_max = y.min() - 10, y.max() + 10
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def meanvec(liste):
    """
    # Finds the mean values for a list consists of n samples
    # num of elements in mean = num of features
    # returns mean vector (2x1)
    """
    mean = [0 for category in liste[0]]
    for x in liste:
        for i in range(len(x)):
            mean[i] += float(x[i])/len(liste)
    # for i in range(len(mean)):
    #     mean[i] /= len(liste)
    return mean


def covariance(liste):
    """
    # Finds the covariance matrix of a list consists of n samples
    """
    mean = meanvec(liste)
    mt_covariance = [[0 for i in mean] for m in mean];
    for sample in liste:
        element = [0 for i in range(len(sample))]
        for i in range(len(sample)):
            element[i] += float(sample[i]) - mean[i]
        for i in range(len(element)):
            for j in range(len(element)):
                mt_covariance[i][j] += (element[i] * element[j])/len(liste)
    return mt_covariance



def discriminant(sample, class_ix, mt_cov, mean):
    """
    # returns a number (result of the discrimination function)
    # returns a number (result of the discrimination function)
    """
    temp = [0.0 for i in range(len(sample))]
    var = 1.0

    for i in range(len(temp)):
        e = 0.0
        for j in range(len(sample)):
            e += float(sample[j]) * W(mt_cov)[j][i]
        temp[i] = e
        var += e * float(sample[i])
        var += w(mt_cov, mean)[i] * float(sample[i])
    var += w0(mt_cov, mean, class_ix)
    var = round(var, 2)
    return var


def W(mt_cov):
    """
    :param mt_cov:      2x2 covariance matrix
    :return:            2x2 arr
    """

    temp = [[0.0 for i in range(len(mt_cov))] for j in range(len(mt_cov))]
    for i in range(len(mt_cov)):
        for j in range(len(mt_cov)):
            temp[i][j] = -0.5 * mt_cov[i][j]
    return temp


def w(mt_cov, mean):
    """
    returns the result of multiplication inverse covariance * mean

    :param mt_cov:      2x2 covariance matrix
    :param mean:        2x1 mean vector
    :return:            2x1 arr
    """
    inv = np.linalg.inv(mt_cov)
    temp = [0 for i in range(len(mean))]
    for i in range(len(inv)):
        e = 0.0
        for j in range(len(inv[i])):
            e += inv[i][j]*mean[j]
        temp[i] = e
    return temp



def w0(mt_cov, mean, index):
    """
    :param mt_cov:  2x2 covariance matrix
    :param mean:    2x1 mean vector
    :param index:   class index
    :return:        float
    """

    localw = w(mt_cov, mean)
    var1 = 0
    var2 = math.log(np.linalg.det(mt_cov))
    for i in range(len(mean)):
        var1 += mean[i]*localw[i]
    result = - 0.5 * (var1 + var2) + math.log(prior(index))
    return result


def prior(index):
    # param: index of the class in traindata
    return len(traindata[index]) / total(traindata)


def total(data):
    size = 0
    for datum in data:
        size += len(datum)
    return size


def splitdata(dataset):
    """
    splits the given dataset (txt file) into two equal sets each with size of number of classes

    :param dataset: path to text file
    :return: both training and test data sets (lists)
    """
    test = []
    train = []
    labels = []
    subsets = []
    with open(dataset, "r") as datafile:
        for line in datafile:
            line = line[:-1]
            values = line.split(" ")
            while "" in values:
                values.remove("")
            if values[-1] not in labels:
                labels.append(values[-1])
                subsets.append([])
            subsets[labels.index(values[-1])].append(values[:-1])

    for subset in subsets:
        train.append(subset[:round(len(subset)/2)])
        test.append(subset[round(len(subset)/2):])

    return train, test, labels



def main(data):
    global traindata, testdata, categories
    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, "data/"+data)
    traindata, testdata, categories = splitdata(path)

    # Xk = traindata[n][k] where k, n = 0, 1, 2...
    # k is the number of samples, n is the number of features

    clf = classifier(categories)
    clf.train(traindata)
    error = clf.test(testdata)
    accuracy = "%.2f" % ((1-(error/total(testdata))) * 100)
    print("Data: ./data/", data, "> Accuracy (%):", accuracy, "[Errors: %d" % error, "/%d]" % total(testdata))
    clf.printclf()
    clf.plotclf(total(traindata), data)


