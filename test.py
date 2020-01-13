import unittest
import numpy as np
from Layer import Layer
from Metric import euclideanMetric
from Network import Network

class TestLayer(unittest.TestCase) :

    def test_layer(self):
        layer = Layer()
        examples = [[1,2,3],[4,5,6]]
        
        a = np.array(examples[0])
        b = np.array(examples[1])
    
        layer.addExamples(examples)

        value=np.array([2,3,4])
        e0 = euclideanMetric(a, value)
        e1 = euclideanMetric(b, value)
        
        ans = layer.evaluate(value, euclideanMetric)
        
        print('ans', ans, e0, e1)
        
        self.assertEqual(ans[0], e0)
        self.assertEqual(ans[1], e1)

    def test_network(self) :

        network = Network(metric=euclideanMetric)
        network.constructNextLayer([[1,2,3],[4,5,6]])
        network.constructNextLayer([[-1,-2,-3],[2,3,4]])
        network.constructNextLayer([[5,7,8],[0,10,11],[1,2,3]],['c1','c2','c3'])

        network.bestClass([1,2,3])


if __name__ == '__main__':
    unittest.main()