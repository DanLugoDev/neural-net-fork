import { zipWith, add, multiply, subtract, all, is, transpose } from 'ramda'

export type FillFn = () => number

/**
 * Always returns zero.
 * @returns {number} zero
 */
export const zero : FillFn = () => 0

/**
 * A Vec is an array of numbers
 */
export type Vec = number[]

/**
 * A Mat is an array of Vecs, all of the same length. The vectors represent
 * rows.
 */
export type Mat = Vec[]

/**
 * Both Mates and Vecs are Tensors
 */
export type Tensor = Vec | Mat

/**
 * Create a new Vec.
 *
 * @param {number} length length of the resulting Vec
 * @param {FillFn=} fillFn returns a number to fill each index with
 */
export const newVec =
  (length : number, fillFn : FillFn = zero) : Vec =>
    (new Array(length))
      .fill(null)
      .map(fillFn)



/**
 * Vectorize a number, basically puts it in a Vec in a provided index and
 * all other values will be zero.
 *
 * Vecize(4, 1, 99) => [0, 99, 0, 0]
 *
 * @param {number} len length of the resulting Vec
 * @param {number} idx index where the number will be positioned
 */
export const Vectorize =
  (len : number, idx : number, value : number) : Vec =>
    newVec(len, zero)
      .map((_,i) => i == idx ? value : 0)

/**
 * Check the Tensor provided is a valid Vec.
 *
 * @param {Tensor} tensor The tensor to be checked if it's a Vec
 */
export const isVec =
  (tensor : Tensor) : tensor is Vec =>
    all(is(Number))(tensor)


/**
 * Check the Tensor provided is a valid Mat.
 *
 * @param {Tensor} tensor The tensor to be checked if it's a Mat
 */
export const isMat =
  (tensor : Tensor) : tensor is Mat =>
    all(is(isVec))(tensor)



export const newMat =
  (rows : number, cols : number, fillFn : FillFn = zero) : Mat =>
    (new Array(rows))
      .fill(newVec(cols, fillFn))






////////////////////////////////////////////////////////////////////////////////
// VECTOR OPERATIONS
////////////////////////////////////////////////////////////////////////////////

export const vecPlusVec = (v : Vec, w : Vec) : Vec => {
  if (!isVec(v) || !isVec(w)) {
    throw new TypeError('Expected 2 Vectors as arguments')
  }
  return zipWith<number, number, number>(add, v, w)
}


export const vecMinusVec = (v : Vec, w : Vec) : Vec => {
  if (!isVec(v) || !isVec(w)) {
    throw new TypeError('Expected Vecs as arguments')
  }
  return zipWith<number, number, number>(subtract, v, w)
}


export const scalarTimesVec = (s : number, v : Vec) : Vec => {
  if ((typeof s != 'number') || !isVec(v)) {
    throw new TypeError('Expected a number and a vector as arguments')
  }
  return v.map(elm => elm * s)
}


export const hadamard = (v : Vec, w : Vec) : Vec => {
  if (!isVec(v) || !isVec(w)) {
    throw new TypeError('Expected Vecs as arguments')
  }
  if (v.length != w.length) throw new TypeError('Vecs should be same length')
  return zipWith<number, number, number>(multiply, v, w)
}


export const schur = hadamard


export const dot = (v : Vec, w : Vec) : number => {
  if (!isVec(v) || !isVec(w)) {
    throw new TypeError('Expected Vecs as arguments')
  }
  return hadamard(v, w).reduce(add, 0)
}


////////////////////////////////////////////////////////////////////////////////
// VECTOR AND MATRIX OPERATIONS
////////////////////////////////////////////////////////////////////////////////
export const vecTimesMat = (v : Vec, m : Mat) : Vec => {
  if (!isVec(v) || !isMat(m)) {
    throw new TypeError('Expected a vector and a matrix as arguments')
  }

  return m.map(row => dot(row, v))
}

////////////////////////////////////////////////////////////////////////////////
// MATRIX OPERATIONS
////////////////////////////////////////////////////////////////////////////////

export const matPlusMat = (m : Mat, n : Mat) : Mat => {
  if (!isMat(m) || !isMat(n)) {
    throw new TypeError('Expected two matrices as arguments')
  }
  return zipWith<Vec, Vec, Vec>(vecPlusVec, m, n)
}


export const matMinusMat = (m : Mat, n : Mat) : Mat => {
  if (!isMat(m) || !isMat(n)) {
    throw new TypeError('Expected two matrices as arguments')
  }
  return zipWith<Vec, Vec, Vec>(vecMinusVec, m, n)
}


export const scalarTimesMat = (s : number, m : Mat) : Mat => {
  if ((typeof s != 'number') || !isMat(m)) {
    throw new TypeError('Expected a number and a matrix as arguments')
  }
  return m.map( row => scalarTimesVec(s, row) )
}


export const matTimesMat = (m : Mat, n : Mat) : Mat => {
  if (!isMat(m) || !isMat(n)) {
    throw new TypeError('Expected two matrices as arguments')
  }

  // can only multiply two matrices if the numer of columns of the first matrix
  // are the same as the number of rows in the second matrix

  const cols : number = transpose(m).length
  const rows : number = n.length

  if (cols != rows) {
    throw new TypeError(
      'can only multiply two matrices if the number of columns of the first' +
      ' matrix are the same as the number of rows in the second matrix')
  }

  return m
    .map(row => vecTimesMat(row, n))
}
