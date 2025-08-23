import Network from '@app/Network'

import {loadDataWrapper} from '@app/Data'

const [ trainingData, , testData ] = loadDataWrapper()


const net = new Network([784, 30, 10])

net.SGD(trainingData, 30, 10, 3.0, testData)
