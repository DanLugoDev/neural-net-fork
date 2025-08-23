"""
File in charge of converting mnist.pkl.gz to mnist.json for convenient
handling in typescript.

Caution: When the original data has only a single input vectors - digits out
tuple the digit out array gets converted to just a number, will cause errors
in the typescript part of the app

Made for python 3 or python 2
"""

import json
import gzip

try:
    import cPickle as pickle
except ImportError:
    import pickle

f = gzip.open('data/mnist.pkl.gz', 'rb')


training_data, validation_data, test_data = pickle.load(f, encoding="latin1")

f.close()

data = { \
    "training_data": [ training_data[0].tolist() , training_data[1].tolist() ] , \
    "validation_data": [ validation_data[0].tolist() , validation_data[1].tolist() ] , \
    "test_data" : [ test_data[0].tolist() , test_data[1].tolist() ] \
}

with open('data/mnist.json', 'w') as outfile:
    json.dump(data, outfile, indent=2)
