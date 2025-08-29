import * as fs from 'fs'
import { transpose, zipWith, splitEvery, range,
         tail, init, head, last }               from 'ramda'
import { shuffle }                               from 'underscore'

import { FillFn, Vec, newVec, isVec, Mat,
         newMat, vecPlusVec, scalarTimesVec,
         scalarTimesMat, matPlusMat,
         matMinusMat, vecMinusVec, vecTimesMat,
         hadamard, dot }                        from '@app/Algebra'
import { InOut, InDigit }                       from '@app/Data'
import { sigmoid, sigmoidPrime }                from '@app/Math'

import * as gaussian from 'gaussian'
// mean 0 variance 1
const distribution = gaussian(0, 1)
/**
 * Take a random sample using inverse transform sampling method.
 * @returns the random sample
 */
const sample = () : number => distribution.ppf(Math.random())






export default class Network {
  /**
   * Dan: Create an array of Vectors, representing the biases of each
   * hidden layer + the output layer of the neural network.
   *
   * @param sizes Vector representing the size of each layer of the
   * network including the input layer (biases wont be created for this layer).
   * @param fillFn (optional) function returning a number to
   * initialize each bias
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
   * @param sizes the sizes of all layers in the network, including the
   * input layer's size
   * @param fillFn (optional) function returning a number to
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

   */
  private numLayers : number


  /**
   * One vector for each layer (variable length), one bias for each neuron
   */
  private biases : Vec[]


  /**
   * One matrix for each layer (variable length),
   * inside each matrix: one vector for each neuron
   * (all vectors same length, vector length same as amount of neurons of
   * the previous layer)
   */
  private weights : Mat[]


  /**
   * A vector with the sizes of each layer.
   */
  private sizes : Vec


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
    if (sizes.length < 3) {
      throw new RangeError('The layer should have at least 3 layers, one input layer, one hidden layer and one output layer')
    }
    this.sizes = sizes
    this.numLayers = this.sizes.length
    this.biases = Network.spawnBiases(this.sizes, sample)
    this.weights = Network.spawnWeights(this.sizes, sample)
  }


  /**
   * Return the output of the network given an input.
   *
   * @param input Vector to the neural network
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
   * @param trainingData List of tuples "[x, y]" representing the
   * training inputs and the desired outputs.
   * @param epochs
   * @param miniBatchSize
   * @param eta The learning rate.
   * @param testData If provided then the network will be evaluated
   * against the test data after each epoch, and partial progress printed out.
   * This is useful for tracking progress, but slows things down substantially.
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

      miniBatches.forEach((miniBatch, i, _) => {
        console.log(`Processing mini-batch ${i+1} out of ${_.length}`)
        this.updateMiniBatch(miniBatch, eta)
      })

      if (testData) {
        console.log(`Epoch ${i}: ${this.evaluate(testData)} numbers correctly identified out of ${testData.length} `)
      } else {
        console.log(`Epoch ${i} complete`)
      }
    })
  }


  /**
   * Update the network's weights and biases by applying gradient descent using
   * back-propagation to a single mini batch.
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
    let secondToLastAct : Vec
    let lastZ : Vec | undefined

    [ deltaNablaB , deltaNablaW ] =
      [ Network.spawnBiases(this.sizes), Network.spawnWeights(this.sizes) ]

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

    secondToLastAct = activations[activations.length -2]
    deltaNablaW[deltaNablaW.length - 1] =
      delta.map(deltaB => scalarTimesVec(deltaB, secondToLastAct))
    /**
     * Note that the variable l in the loop below is used a little
     * differently to the notation in Chapter 2 of the book.  Here,
     * l = 1 means the last layer of neurons, l = 2 is the
     * second-last layer, and so on.  It's a renumbering of the
     * scheme in the book, used here to take advantage of the fact
     * that Python can use negative indices in lists.
     */
    /**
     * Dan: We'll use arr[arr.length - l] as the equivalent except for
     * l = 0, however, l here wont ever be zero.
     *
     * Also I took the liberty to change the variable here to k, as l looks a
     * lot like 1 (one) so the code gets confusing
     */
    //
    range(2, this.numLayers).forEach(k => {
      const z = zs[zs.length - k]
      const sp = z.map(sigmoidPrime)
      const deltaedWeights : Vec = transpose(this.weights[this.weights.length -k + 1]).map(w => dot(w, delta))
      delta = hadamard(deltaedWeights, sp)
      deltaNablaB[deltaNablaB.length - k] = delta
      deltaNablaW[deltaNablaW.length - k] = delta.map(elm => scalarTimesVec(elm, activations[activations.length - k - 1]))
    })

    return [ deltaNablaB , deltaNablaW ]
  }

  ingress(fileName: string) {
    const json = JSON.parse(fs.readFileSync(fileName, 'utf8'))
    const biases = json.biases as number[][]
    const weights = json.weights as number[][][]
    this.biases = biases
    this.weights = weights

    this.sizes = [784, ...weights.map(b => b.length)]
    this.numLayers = this.sizes.length
  }

  dump() {
    const jsContent = `
      export const weights = ${JSON.stringify(this.weights, null, 2)};
      export const biases = ${JSON.stringify(this.biases, null, 2)};
      `.trim();

    const layerString = this.biases
      ? [this.biases[0].length, ...this.biases.map(b => b.length)].join('-')
      : 'unknown-layers';
    fs.writeFileSync(
      `net-${layerString}-${new Date().getTime()}.js`,
      jsContent
    );

    fs.writeFileSync(
      `net-${layerString}-${new Date().getTime()}.json`,
      {
        biases: this.biases,
        weights: this.weights,
      },
    );
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

      console.log(`Evaluating vector, expecting ${expectedDigit}`)

      const networkOutput : Vec = this.feedforward(inputVector)

     

      const biggestActivation = Math.max(...networkOutput)

      const digitPredicted : number = networkOutput.indexOf(biggestActivation)
      console.log(`Digit predicted: ${digitPredicted}`)

      testResults.push(digitPredicted == expectedDigit)
    }


    return testResults.filter(Boolean).length
  }
}
