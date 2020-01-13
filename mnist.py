import numpy as np
from Layer import Layer
from Metric import euclideanMetric
from Metric import radialMetric
from Network import Network
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

print('X',X.shape,'y',y.shape)
print(y[1:50])

train_samples = 1000
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_samples, test_size=100)

network = Network(metric=euclideanMetric)

layers = 10

batch = 1000 #int(train_samples/layers)
for i in range(0, layers) :
    #network.constructNextLayer(X_train[i*batch:(i+1)*batch],y_train[i*batch:(i+1)*batch])
    network.constructNextLayer(X_train,y_train)

for i in range(0, layers) :
    ans = network.bestClass(X_test, reverse=True, layerIndex=i)
    score = accuracy_score(y_test,ans)
    print('layer', i, 'score', score)
