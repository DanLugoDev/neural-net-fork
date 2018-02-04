import numpy as np
import json

data = json.load(open('test-parameters.json'))

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoidPrime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def costDx(output_activations, y):
    return (output_activations-y)


entry = data['entry']
sizes = data['sizes']
numLayers = data['numLayers']
biases = data['biases']
weights = data['weights']
expectedOut = data['expectedOut']


npbiases = [np.random.randn(y, 1) for y in sizes[1:]]
npweights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]



weights = list(map(lambda w : np.array(w), weights))

newBiases = []

entry = list(map(lambda e : [e], entry))

expectedOut = list(map(lambda e : [e], expectedOut))

for i in range(0, len(biases)):
    elm = np.zeros((len(biases[i]), 1))
    biasArr = []
    for j in range(0, len(biases[i])):
        elm[j] = biases[i][j]
    newBiases.append(elm)

biases = newBiases

for w, npw in zip(weights, npweights):
    assert w.shape == npw.shape

for b, npb in zip(biases, npbiases):
    assert b.shape == npb.shape



nabla_b = [np.zeros(b.shape) for b in npbiases]
nabla_w = [np.zeros(w.shape) for w in npweights]

x = np.array(entry)
activation = x
activations = [x]
zs = []

for b, w in zip(biases, weights):
    z = np.dot(w, activation) + b
    zs.append(z)
    activation = sigmoid(z)
    activations.append(activation)


delta = (activations[-1] - expectedOut) * sigmoidPrime(zs[-1])
nabla_b[-1] = delta






nabla_w[-1] = np.dot(delta, activations[-2].transpose())

for l in range(2, numLayers):
    z = zs[-l]
    sp = sigmoidPrime(z)
    print('old delta')
    print(delta)
    print('\n')
    print('weights[-l+1]')
    print(weights[-l+1])
    print('\n')
    print('weights[-l+1].transpose()')
    print(weights[-l+1].transpose())
    print('\n')
    print('zs[-l]')
    print(z)
    print('\n')
    print('sp')
    print(sp)
    print('\n')
    print('np.dot(weights[-l+1].transpose(), delta)')
    print(np.dot(weights[-l+1].transpose(), delta))
    print('\n')
    delta = np.dot(weights[-l+1].transpose(), delta) * sp
    print('new delta')
    print(delta)
    print('\n')
    nabla_b[-l] = delta
    print('activations')
    print(activations)
    nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    print('np.dot(delta, activations[-l-1].transpose())')
    print(np.dot(delta, activations[-l-1].transpose()))
    print('\n\n********\n\n')
