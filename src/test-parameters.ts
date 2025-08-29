import * as fs from 'fs'
import { zipWith, tail, init } from 'ramda'

import { FillFn, Vec, newVec, Mat, newMat} from '@app/Algebra'


const gaussian = require('gaussian')
// mean 0 variance 1
const distribution = gaussian(0, 1)
/**
* Take a random sample using inverse transform sampling method.
* @returns the random sample
*/
const sample : FillFn = () => distribution.ppf(Math.random())

const
  sizes : Vec = [2,3,3,1],
  numLayers : number = sizes.length

let biases : Vec[] = tail(sizes).map( size => newVec(size, sample) )

let weights : Mat[] = zipWith<number,number,Mat>(
  (s1, s2) => newMat(s1, s2, sample),
  tail(sizes),
  init(sizes)
)



let entry : Vec = [0.7, 0.3]

if (entry.length !== sizes[0]) throw new Error()

let expectedOut : Vec = [0.378756]

if (expectedOut.length !== sizes[sizes.length - 1]) throw new Error()

let testParameters = {
  entry,
  sizes,
  numLayers,
  biases,
  weights,
  expectedOut
}

fs.exists('test-parameters.json', (exists : boolean) => {
  if (exists) {
    fs.unlinkSync('test-parameters.json')
  }

  const json = JSON.stringify(testParameters)

  fs.writeFileSync('test-parameters.json', json)
})
