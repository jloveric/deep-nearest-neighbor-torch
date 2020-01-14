import unittest
import numpy as np
from Layer import Layer
from Metric import euclideanMetric
from Metric import radialMetric
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

        inList = [[1,2,3],[4,5,6]]
        network.constructNextLayer([[1,2,3],[4,5,6]])
        network.constructNextLayer([[1,2,3],[4,5,6]])
        network.constructNextLayer([[1,2,3],[4,5,6]],['c1','c2','c3'])

        evalList = [[1,2,3],[4,5,6]]
        res = network.evaluateList(evalList)
        print('res', res)

        ans = network.probability(evalList, reverse=True)
        print('ans', ans)
        ans = network.bestClass(evalList, reverse=True)
        print('ans', ans)
        self.assertEqual(ans[0],'c1')
        self.assertEqual(ans[1],'c2')

    def test_classes(self) :
        network = Network(metric=euclideanMetric)

        inList = [[1,2,3],[4,5,6]]
        network.constructNextLayer(np.array([[1,2,3],[4,5,6],[7,8,9]]),['c1','c2','c3'])
        network.constructNextLayer(np.array([[1,2,3],[4,5,6]]),['c1','c2'])
        network.constructNextLayer(np.array([[1,2,3],[4,5,6]]),['c1','c2'])

        for i in range(0,3) :
            ans = network.bestClass(inList, reverse=True, layerIndex=i)
            print('best class', ans)


if __name__ == '__main__':
    unittest.main()