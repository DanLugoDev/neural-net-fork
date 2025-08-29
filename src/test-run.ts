import Network from '@app/Network'

import {loadDataWrapper} from '@app/Data'

const [ _trainingData, , _testData ] = loadDataWrapper()

const testData = _testData.slice(0, 50)

const trainingData = _trainingData.slice(0, 50)

console.log(`Training data length: ${trainingData.length}`)
console.log(`Test data length: ${testData.length}`)

const net = new Network([784, 90,15,90, 10])

net.SGD(trainingData, 1, 10, 3.0)

net.dump()

// net.ingress('data/demo-net.json')

// const result = net.evaluate(testData)
// console.log(`Got ${result} correct out of ${testData.length}`)