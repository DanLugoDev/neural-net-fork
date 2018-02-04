

export const sigmoid = (n : number) : number => 1 / ( 1 + Math.exp(-n) )



// derivative of the sigmoid function
export const sigmoidPrime = (z : number) : number => sigmoid(z) * ( 1 - sigmoid(z) )



export const softplus = (n : number) : number => {
  return Math.log(1 + Math.exp(n))
}




export const rnd = (low : number = 0, high : number = 1) : number => {
  return Math.floor((Math.random() * (low - high)) + low)
}





export const gaussRnd = () => (Math.random() - 0.5) * 10
