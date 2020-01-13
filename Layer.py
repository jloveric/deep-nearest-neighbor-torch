import numpy as np

class Layer :

    def __init__(self) :
        self.data = []

    def addExample(self, example) :
        print('Not currently implemented')
        pass

    def addExamples(self, examples) :
        self.data = np.array(examples)
    
    def evaluate(self, value, metric) :
        result = []

        for x in self.data :
            a = metric(x, value)
            result.append(a)

        return np.array(result)
        
