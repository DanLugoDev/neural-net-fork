import Network from '@app/Network'

import {loadDataWrapper} from '@app/Data'

const [ _trainingData, , _testData ] = loadDataWrapper()

const testData = _testData.slice(0, 100)

const trainingData = _trainingData.slice(0, 100)

console.log(`Training data length: ${trainingData.length}`)
console.log(`Test data length: ${testData.length}`)

const net = new Network([784, 16, 16, 10])

// net.SGD(trainingData, 30, 10, 3.0)

// net.dump()

net.ingress('data/demo-net.json')

const result = net.evaluate(testData)

console.log(`Got ${result} correct out of ${testData.length}`)