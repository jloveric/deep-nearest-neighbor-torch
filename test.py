import unittest
import numpy as np
from Layer import Layer

def simpleMetric(x,y) :
    a = x-y
    return np.linalg.norm(a)

class TestLayer(unittest.TestCase) :

    def test_layer(self):
        layer = Layer()
        examples = [[1,2,3],[4,5,6]]
        
        a = np.array(examples[0])
        b = np.array(examples[1])
    
        layer.addExamples(examples)

        value=np.array([2,3,4])
        e0 = simpleMetric(a, value)
        e1 = simpleMetric(b, value)
        
        ans = layer.evaluate(value, simpleMetric)
        
        print('ans', ans, e0, e1)
        
        self.assertEqual(ans[0], e0)
        self.assertEqual(ans[1], e1)

if __name__ == '__main__':
    unittest.main()