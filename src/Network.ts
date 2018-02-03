import { zipWith, splitEvery, range, tail, init, head, last } from 'ramda'



import { sigmoid, sigmoidPrime }                from '@app/Math'
import { InOut, InDigit }                       from '@app/Data'
import { shuffle }                              from 'underscore'
import { FillFn, Vec, newVec, isVec, Mat,
         newMat, vecPlusVec, scalarTimesVec,
         scalarTimesMat, matPlusMat,
         matMinusMat, vecMinusVec, vecTimesMat,
         hadamard }                             from '@app/Algebra'

const gaussian = require('gaussian')
// mean 0 variance 1
const distribution = gaussian(0, 1)
/**
 * Take a random sample using inverse transform sampling method.
 * @returns {number} the random sample
 */
const sample = () : number => distribution.ppf(Math.random())













export default class Network {
  /**
   * Dan: Create an array of Vectors, representing the biases of each
   * hidden layer + the output layer of the neural network.
   *
   * @param {Vec} sizes Vector representing the size of each layer of the
   * network including the input layer (biases wont be created for this layer).
   * @param {FillFn=} fillFn (optional) function returning a number to
   * initialize each bias
   * @returns {Vec[]}
   */
  private static spawnBiases (sizes : Vec, fillFn? : FillFn) : Vec[] {
    // Dan: ignore the first element of the sizes vector as this is the
    // size of the input layer, no biases should be set for the input layer.
    return tail(sizes)
      .map( size => newVec(size, fillFn) )
  }



  /**
   * Dan: Create an array of Matrixes, each one representing the weights
   * connecting a layer to the previous one.
   * No weights are set for the input layer.
   * Dan: To get the weights:
   *
   * init(vector) => all but the last element of the vector
   * e.g. init([a,b,c]) => [a,b]
   * tail(vector) => all but the first elemnet of the vector
   * e.g. tail([a,b,c]) => [b,c]
   *
   * zip the tail and init of the sizes vector (IN THAT ORDER) the two numbers
   * are the dimensions of the weights matrix for each layer.
   *
   * @param {Vec} sizes the sizes of all layers in the network, including the
   * input layer's size
   * @param {FillFn} fillFn (optional) function returning a number to
   * initialize each weight
   */
  private static spawnWeights (sizes : Vec, fillFn? : FillFn) : Mat[] {
    return zipWith<number,number,Mat>(
      (s1, s2) => newMat(s1, s2, fillFn),
      tail(sizes),
      init(sizes)
    )
  }




  /**
   * The number of layers in the neural network
   * @type {number}
   */
  private readonly numLayers : number


  /**
   * One vector for each layer (variable length), one bias for each neuron
   * @type {Vec[]}
   */
  private biases : Vec[]


  /**
   * One matrix for each layer (variable length),
   * inside each matrix: one vector for each neuron
   * (all vectors same length, vector length same as amount of neurons of
   * the previous layer)
   * @type {Mat[]}
   */
  private weights : Mat[]


  /**
   * A vector with the sizes of each layer.
   * @type {Vec}
   */
  private readonly sizes : Vec


  /**
   * mnielsen: The list ``sizes`` contains the number of neurons in the
   * respective layers of the network.  For example, if the list
   * was [2, 3, 1] then it would be a three-layer network, with the
   * first layer containing 2 neurons, the second layer 3 neurons,
   * and the third layer 1 neuron.  The biases and weights for the
   * network are initialized randomly, using a Gaussian
   * distribution with mean 0, and variance 1.  Note that the first
   * layer is assumed to be an input layer, and by convention we
   * won't set any biases for those neurons, since biases are only
   * ever used in computing the outputs from later layers.
   */
  constructor (sizes : Vec) {
    this.sizes = sizes
    this.numLayers = this.sizes.length
    this.biases = Network.spawnBiases(this.sizes, sample)
    this.weights = Network.spawnWeights(this.sizes, sample)
  }


  /**
   * Return the output of the network given an input.
   *
   * @param {Vec} input Vector to the neural network
   */
  feedforward (input : Vec) : Vec {
    if (!isVec(input)) throw new TypeError('expected vector as argument')

    if (input.length != this.sizes[0]) {
      throw new TypeError(
        'Vector length incorrect, should be the same length as the input layer'
      )
    }

    let activation : Vec = input

    zipWith<Vec, Mat, void>(
      (layerBiases: Vec, layerWeights: Mat) => {
        let weighted : Vec = vecTimesMat(activation, layerWeights)
        let z : Vec = vecPlusVec(weighted, layerBiases)
        let squashed : Vec = z.map(sigmoid)

        activation = squashed
      },
      this.biases,
      this.weights
    )

    return activation
  }


  /**
   * Train the neural network using mini-batch stochastic gradient descent.
   *
   * @param {InOut[]} trainingData List of tuples "[x, y]" representing the
   * training inputs and the desired outputs.
   * @param {number} epochs
   * @param {number} miniBatchSize
   * @param {number} eta
   * @param {InDigit[]=} testData If provided then the network will be eavaluated
   * against the test data after each epoch, and partial progress printed out.
   * This is useful for tracking progress, but slows things down subtantially.
   */
  SGD (
    trainingData : InOut[],
    epochs : number,
    miniBatchSize : number,
    eta : number,
    testData? : InDigit[]
  ) : void {
    range(0, epochs).forEach(i => {
      const shuffled : InOut[] = shuffle(trainingData)
      const miniBatches : InOut[][] = splitEvery(miniBatchSize, shuffled)

      for (const miniBatch of miniBatches) {
        this.updateMiniBatch(miniBatch, eta)
      }

      if (testData) {
        console.log(`Epoch ${i}: ${this.evaluate(testData)} correct / ${testData.length} `)
      } else {
        console.log(`Epoch ${i} complete`)
      }
    })
  }


  /**
   * Update the network's weights and biases by applying gradient descent using
   * backpropagation to a single mini batch.
   * @param miniBatch List of tuples "[x, y]" representing the training
   * inputs and the desired outputs. A subset of all training examples.
   * @param eta The learning rate
   */
  updateMiniBatch (miniBatch : InOut[], eta : number) : void {
    const normalEta : number = eta / miniBatch.length
    const scaleLayerBiasesByEta : (v : Vec) => Vec =
      scalarTimesVec.bind(null, normalEta)
    const scaleLayerWeightsByEta : (m : Mat) => Mat =
      scalarTimesMat.bind(null, normalEta)

    let deltaNablaB : Vec[]
    let deltaNablaW : Mat[]

    [ deltaNablaB, deltaNablaW ] =
      [ Network.spawnBiases(this.sizes) , Network.spawnWeights(this.sizes) ]

    for (const inOut of miniBatch) {
      const [ givenIn , expectedOut ] = inOut
      const [ diffDeltaNablaB , diffDeltaNablaW ] =
        this.backprop(givenIn, expectedOut)

      deltaNablaB = zipWith(vecPlusVec, deltaNablaB, diffDeltaNablaB)
      deltaNablaW = zipWith(matPlusMat, deltaNablaW, diffDeltaNablaW)
    }

    const scaledDeltaNablaB : Vec[] = deltaNablaB.map(scaleLayerBiasesByEta)
    const scaledDeltaNablaW : Mat[] = deltaNablaW.map(scaleLayerWeightsByEta)

    this.biases = zipWith(vecMinusVec, this.biases, scaledDeltaNablaB)

    this.weights = zipWith(matMinusMat, this.weights, scaledDeltaNablaW)
  }


  // Dan: Changed names of variables here to delta__
  // since they should be called delta everywhere
  /**
   * Return a tuple [nabla_b, nabla_w] representing the gradient for the cost
   * function C_x. "nabla_b" and "nabla_w" are layer-by-layer lists of
   * arrays, similar to biases and weights.
   *
   *
   * @param givenIn
   * @param expectedOut
   */
  backprop (givenIn : Vec, expectedOut : Vec) : [ Vec[], Mat[] ] {
    if (givenIn.length != head(this.sizes)) {
      throw new TypeError(
        'givenIn vector incorrect length, should be same length as input layer'
      )
    }

    if (expectedOut.length != last(this.sizes)) {
      throw new TypeError(
        'expectedOut vector incorrect length should be same length as out layer'
      )
    }

    let deltaNablaB : Vec[]
    let deltaNablaW : Mat[]

    [ deltaNablaB , deltaNablaW ] =
      [ Network.spawnBiases(this.sizes), Network.spawnWeights(this.sizes) ]

    // feedforward
    let activation : Vec = givenIn
    // list to store all the activations, layer by layer
    let activations : Vec[] = [givenIn]
    // list to store all the z vectors, layer by layer
    let zs : Vec[] = []
    // Dan: for backward pass
    let delta : Vec
    let costDerivative : Vec
    let lastActivation : Vec | undefined
    let lastZ : Vec | undefined

    zipWith(
      (layerBiases : Vec, layerWeights : Mat) => {
        const weighted : Vec  = vecTimesMat(activation, layerWeights)
        const z : Vec = vecPlusVec(weighted, layerBiases)
        zs.push(z)
        activation = z.map(sigmoid)
        activations.push(activation)
      },
      this.biases,
      this.weights
    )

    // Dan: keep in mind to understand the code: after the zipWith above
    // the last Z in zs and the last activation in activations will be
    // vectors of length equal the output layer's length.
    //
    // Estos vienen siendo output y outputZ por eso se pueden operar
    // safely with the expectedOut vector, and the delta vector can be
    // safely assigned to the last index of the deltaNablaB
    //
    // Last index of deltaNablaW is a matrix with number of rows equal to the
    // number of neurons in the previous-to-last layer

    // backwards pass
    lastActivation = last(activations)
    lastZ = last(zs)
    if (typeof lastActivation == 'undefined') {
      throw new ReferenceError('lastActivation is undefined. Check you set a correct number of layers or else there\'s a bug in the app.')
    }
    if (typeof lastZ == 'undefined') {
      throw new ReferenceError('lastZ is undefined. Check you set a correct number of layers or else there\'s a bug in the app.')
    }

    costDerivative = vecMinusVec(lastActivation, expectedOut)
    delta = hadamard(costDerivative, lastZ.map(sigmoidPrime))
    deltaNablaB[deltaNablaB.length - 1] = delta


    //deltaNablaW[deltaNablaW.length - 1] = vecTimesMat()
    /**
     * Note that the variable l in the loop below is used a little
     * differently to the notation in Chapter 2 of the book.  Here,
     * l = 1 means the last layer of neurons, l = 2 is the
     * second-last layer, and so on.  It's a renumbering of the
     * scheme in the book, used here to take advantage of the fact
     * that Python can use negative indices in lists.
     */

    range(2, this.numLayers).forEach(l => {
      const z = zs[zs.length - l]
      const sp = z.map(sigmoidPrime)

      delta = vecTimes(
        sp,

      )

      deltaNablaB[deltaNablaB.length - l] = delta
      deltaNablaW[deltaNablaW.length - l] =
    })

    return [ deltaNablaB , deltaNablaW ]
  }





  /**
   * Returns the number of test inputs for which the neural network outputs the
   * correct result. NOTE that the neural network's output is assumed to be the
   * index of whichever neuron in the final layer has the highest activation.
   * @param testData
   */
  evaluate (testData : InDigit[]) : number {
    const testResults : boolean[] = []


    for (const testExample of testData) {
      const [ inputVector , expectedDigit ] = testExample
      const networkOutput : Vec = this.feedforward(inputVector)

      const digitPredicted : number = networkOutput.reduce(
        (max : number, value : number) => Math.max(max, value),
        0
      )

      testResults.push(digitPredicted == expectedDigit)
    }


    return testResults.reduce(
      // Number(true) => 1 , Number(false) => 0
      (counter : number, result : boolean) => counter + Number(result),
      0
    )
  }
}
