from Layer import Layer
import numpy as np

class Network : 

    def __init__(self, metric) :
        self.layer = []
        self.metric = metric
    
    #Construct the next layer given examples
    def constructNextLayer(self, examples, classes=None) :
        output = self.evaluateList(examples)
        layer = Layer()
        layer.addExamples(output, classes)
        self.layer.append(layer)

    def evaluate(self, value) :
        out = value
        for tLayer in self.layer :

            out = tLayer.evaluate(out, self.metric)
        
        return out

    def evaluateList(self, values) :
        out = values
        for tLayer in self.layer :
            out = tLayer.evaluateList(out, self.metric)
        
        return out

    def bestClass(self, values, layerIndex=-1) :
        ans = self.evaluateList(values)
        argmax = []
        lastLayer = self.layer[layerIndex]

        for i in ans :
            argmax.append(lastLayer.classes[np.argmax(i)])