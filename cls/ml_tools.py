import theano
import theano.tensor as T
import numpy as np

class FlatImages(object):
    def __init__(self):
        self.X=T.matrix('x')
        self.y=T.lvector('y')

    def get_vars(self):
        return [self.X,self.y]

class LayerModel(object):
    def __init__(self,W,b):
        self.W=W
        self.b=b

    def get_params(self):
        return [self.W,self.b]

class Classifier(object):
    def __init__(self,free_vars,model,train,test):
        self.free_vars=free_vars
        self.model=model
        self.test=test
        self.train=train

def create_layer(shape):
    W=init_weights(shape[0],shape[1])
    b=init_bias(shape[1])
    return LayerModel(W,b)

def init_weights(n_in,n_out):
    init_value=get_zeros((n_in,n_out))
    return theano.shared(value=init_value,name='W',borrow=True)

def init_bias(n_out):
    init_value=get_zeros((n_out,))
    return theano.shared(value=init_value,name='b',borrow=True)

def get_zeros(shape):
    return np.zeros(shape,dtype=theano.config.floatX)
