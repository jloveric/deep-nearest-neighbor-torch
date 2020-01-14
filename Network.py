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

    def evaluate(self, value, layerIndex=-1) :
        out = value

        final = len(self.layer)
        if(layerIndex!=-1) :
            final = layerIndex+1

        for i in range(0,final) :
            tLayer = self.layer[i]
            out = tLayer.evaluate(out, self.metric)
        
        return out

    def evaluateList(self, values, layerIndex=-1) :
        out = values
        
        final = len(self.layer)
        if(layerIndex!=-1) :
            final = layerIndex+1

        for i in range(0, final) :
            out = self.layer[i].evaluateList(out, self.metric)
        
        return out
    
    '''
    Compute the "probability" of each class.  Right now it assumes
    the values are always positive as outputs of each layer
    '''
    def probability(self, values, layerIndex=-1, reverse=False) :
        #print('values.shape', values.shape)
        ans = self.evaluateList(values, layerIndex=layerIndex)
        result = []
        argmax = []

        for i in ans :
            #print('i.shape', i.shape)
            normalized = i/np.sum(i)
            if reverse :
                normalized = 1.0-normalized

            result.append(normalized)

        return result

    #Return the most likely class
    def bestClass(self, values, layerIndex=-1, reverse=False) :
        
        ans = self.probability(values, layerIndex=layerIndex, reverse=reverse)
        argmax = []
        #print('layerIndex', layerIndex,'len layer', len(self.layer))
        lastLayer = self.layer[layerIndex]

        for i in ans :
            #print('lastLayer.classes', lastLayer.classes, len(lastLayer.classes), len(ans),i.shape)
            #print('argmax', np.argmax(i))
            argmax.append(lastLayer.classes[np.argmax(i)])

        return argmax