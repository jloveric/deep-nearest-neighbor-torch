import numpy as np

def euclideanMetric(x,y) :
    a = x-y
    return np.linalg.norm(a)