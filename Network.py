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
    
    '''
    Compute the "probability" of each class.  Right now it assumes
    the values are always positive as outputs of each layer
    '''
    def probability(self, values, layerIndex=-1, reverse=False) :
        
        ans = self.evaluateList(values)
        result = []
        lastLayer = self.layer[layerIndex]
        argmax = []

        for i in ans :
            normalized = i/np.sum(i)
            if reverse :
                normalized = 1.0-normalized

            result.append(normalized)

        return result

    #Return the most likely class
    def bestClass(self, values, layerIndex=-1, reverse=False) :
        ans = self.probability(values, layerIndex=layerIndex, reverse=reverse)
        argmax = []
        lastLayer = self.layer[layerIndex]

        for i in ans :
            argmax.append(lastLayer.classes[np.argmax(i)])

        return argmax