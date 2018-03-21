#
# assignment3
# ML_Assignment3
#
# Created by Nehir Poyraz 21.03.2018
# Copyright © 2018 Nehir Poyraz. All rights reserved.


import os.path


def main():

    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, "data/two_class/data1.txt")
    traindata, testdata = splitdata(path)
    testtotal, traintotal = 0, 0
    for i in range(len(traindata)):
        traintotal += len(traindata[i])
    for i in range(len(testdata)):
        testtotal += len(testdata[i])

    featcount = len(traindata[0][0])
    samplecount = len(traindata[0])
    print("Number of features", featcount)
    print("Number of samples in w0", samplecount)

    print("size of traindata[0][0]", len(traindata[0][0]))


    # for category in traindata:
    #     for i in range(len(category)):
    #         fv.append([])
    #         for j in range(len(category[i])):
    #             fv[i].append([])
    #             fv[i][j] = category[i][j]

    # for i in range(len(fv)):
    #     print(fv[i])
    #     print("number of elements in featurevector%d" % i, len(fv[i]))
    print("Number of samples in training data: ", traintotal)
    print("Number of samples in test data: ", testtotal)

    # print("number of elements in fv", len(fv))


def splitdata(dataset):
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
            # print(values)
            if values[-1] not in labels:
                labels.append(values[-1])
                subsets.append([])
            subsets[labels.index(values[-1])].append(values[:-1])
        # print(subsets)

    for subset in subsets:
        train.append(subset[:round(len(subset)/2)])
        test.append(subset[round(len(subset)/2):])

    return train, test

# A = [1, 2, 3, 4, 5, 6, 7]
# B = []
# C = []
#
# B.append(A[:2])
# C.append(A[round(len(A)/2):])
# print(B)
# print(C)

main()
