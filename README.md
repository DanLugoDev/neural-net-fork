# neural-net-fork

Rewrite of [https://github.com/mnielsen/neural-networks-and-deep-learning](https://github.com/mnielsen/neural-networks-and-deep-learning)

## How to use

- Run `scripts/DataConvert.py` with either python2 or python3. This will create `mnist.json` inside the `data` folder.
- `npm run test` will run all unit tests plus a test run using the training and validation data that was just generated.
- Can also load the `src/Network` module from any consumer, instantiate the class, and operate on it.