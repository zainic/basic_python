import numpy as np
import math
def norm(vector):
    sum = 0
    for i in range(len(vector)):
        sum += abs(vector[i]) ** 2
    p = math.sqrt(sum)
    return p

def mean(vector):
    mu = 0
    for i in range(len(vector)):
        mu += vector[i]
    p = mu/len(vector)
    return p

def gray(image_array):
    a = len(image_array[:][0])
    b = len(image_array[0][:])
    X = np.zeros((a, b, 3))
    for i in range(a):
        for j in range(b):
            N = mean(image_array[i][j][0:3])
            for k in range(3):
                X[i][j][k] = N
    return X

def maximize(image_array):
    a = len(image_array[:][0])
    b = len(image_array[0][:])
    maximum = np.amax(image_array)
    delta = 1 - maximum
    X = np.zeros((a, b, 3))
    for i in range(a):
        for j in range(b):
            for k in range(3):
                new = (image_array[i][j][k]/maximum) * delta
                X[i][j][k] = image_array[i][j][k] + new
    return X

def compressed3Dto2D(image_array):
    a = len(image_array[:][0])
    b = len(image_array[0][:])
    X = np.zeros((a, b))
    for i in range(a):
        for j in range(b):
            X[i][j] = mean(image_array[i][j])
    return X

def TwoD_to_OneD(array):
    a = len(array[:][0])
    b = len(array[0][:])
    n = a * b
    vector = np.zeros(n)
    k = 0
    for i in range(a):
        for j in range(b):
            vector[k] = array[i][j]
            k += 1
    return vector