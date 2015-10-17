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

class RandomNum(object):
    def __init__(self):
        self.rng = np.random.RandomState(123)

    def random_matrix(self,n_x,n_y):
        bound=np.sqrt(6. / (n_x + n_y))
        raw_matrix=self.rng.uniform(-bound,bound,n_x*n_y)
        matrix=np.array(raw_matrix,dtype=theano.config.floatX)
        return np.reshape(matrix,(n_x,n_y))

    def random_vector(self,n_x):
        bound=np.sqrt(6. / n_x)
        raw_vector=self.rng.uniform(-bound,bound,n_x)
        return np.array(raw_vector,dtype=theano.config.floatX)

def create_layer(shape,rand=None):
    W=init_weights(shape[0],shape[1],rand)
    b=init_bias(shape[1],rand)
    return LayerModel(W,b)

def init_weights(n_in,n_out,rand=None):
    if(rand==None):
        init_value=get_zeros((n_in,n_out))
    else:
        init_value=rand.random_matrix(n_in,n_out)
    return theano.shared(value=init_value,name='W',borrow=True)

def init_bias(n_out,rand=None):
    if(rand==None):
        init_value=get_zeros((n_out,))
    else:
        init_value=rand.random_vector(n_out)
    return theano.shared(value=init_value,name='b',borrow=True)

def get_zeros(shape):
    return np.zeros(shape,dtype=theano.config.floatX)
