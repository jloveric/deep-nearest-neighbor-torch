import numpy as np

class Layer :

    def __init__(self) :
        self.data = []
        self.classes = None

    def addExample(self, example) :
        print('Not currently implemented')

    def addExamples(self, examples, classes=None) :
        self.data = np.array(examples)
        self.classes = classes
    
    def evaluate(self, value, metric) :
        result = []

        for x in self.data :
            a = metric(x, value)
            result.append(a)

        return np.array(result)
        
    def evaluateList(self, value, metric) :
        final = []
        for y in value :
            result = []
            for x in self.data :
                a = metric(x, value)
                result.append(a)
            final.append(np.array(result))
        
        return final
