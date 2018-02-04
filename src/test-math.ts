import { zipWith, splitEvery, range, tail, init, head, last, transpose } from 'ramda'

import { sigmoid, sigmoidPrime }                from '@app/Math'
import { InOut, InDigit }                       from '@app/Data'
import { shuffle }                              from 'underscore'
import { FillFn, Vec, newVec, isVec, Mat,
         newMat, vecPlusVec, scalarTimesVec,
         scalarTimesMat, matPlusMat,
         matMinusMat, vecMinusVec, vecTimesMat,
         hadamard, dot }                             from '@app/Algebra'


const fs = require('fs')
const gaussian = require('gaussian')

let entry : Vec
let sizes : Vec
let numLayers : number
let biases : Vec[]
let weights : Mat[]
let expectedOut : Vec

let data = JSON.parse(fs.readFileSync('test-parameters.json'))

entry = data.entry
sizes = data.sizes
numLayers = data.numLayers
biases = data.biases
weights = data.weights
expectedOut = data.expectedOut



let deltaNablaB = tail(sizes).map( size => newVec(size, () => 0) )

let deltaNablaW : Mat[] = zipWith<number,number,Mat>(
  (s1, s2) => newMat(s1, s2, () => 0),
  tail(sizes),
  init(sizes)
)

let x : Vec = entry

let activation : Vec = x

let activations : Vec[] = [x]
let zs : Vec[] = []

zipWith(
  (layerBiases : Vec, layerWeights : Mat) => {
    const weighted : Vec  = vecTimesMat(activation, layerWeights)
    const z : Vec = vecPlusVec(weighted, layerBiases)
    zs.push(z)
    activation = z.map(sigmoid)
    activations.push(activation)
  },
  biases,
  weights
)


let lastActivation = last(activations)
let lastZ = last(zs)
if (typeof lastActivation == 'undefined') {
  throw new ReferenceError('lastActivation is undefined. Check you set a correct number of layers or else there\'s a bug in the app.')
}
if (typeof lastZ == 'undefined') {
  throw new ReferenceError('lastZ is undefined. Check you set a correct number of layers or else there\'s a bug in the app.')
}

let costDerivative = vecMinusVec(lastActivation, expectedOut)
let delta = hadamard(costDerivative, lastZ.map(sigmoidPrime))


deltaNablaB[deltaNablaB.length - 1] = delta





let newLastWeight : Mat = delta.map(deltaB => scalarTimesVec(deltaB, activations[activations.length -2]))

deltaNablaW[deltaNablaW.length -1] = newLastWeight

range(2, numLayers).forEach(k => {
  const z = zs[zs.length - k]
  const sp = z.map(sigmoidPrime)
  const deltaedWeights : Vec = transpose(weights[weights.length -k + 1]).map(w => dot(w, delta))
  delta = hadamard(deltaedWeights, sp)
  deltaNablaB[deltaNablaB.length - k] = delta
  deltaNablaW[deltaNablaW.length - k] = delta.map(elm => scalarTimesVec(elm, activations[activations.length - k - 1]))
})
