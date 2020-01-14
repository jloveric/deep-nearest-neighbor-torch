import numpy as np

def euclideanMetric(x,y) :
    a = x-y
    return np.linalg.norm(a)

def sinDistance(x,y) :
    a = 1-np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
    return a

def radialMetric(x,y) :
    a = x-y
    return np.exp(-np.linalg.norm(a))