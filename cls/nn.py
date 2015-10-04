import load,logit
import numpy as np
import theano
import theano.tensor as T
import imp
utils =imp.load_source("utils","/home/user/cf/conv_frames/utils.py")

class HiddenLayer(object):
    def __init__(self,input_data, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
 
        self.input_data = input_data
        self.__init_rng()
        self.__init_W(n_in,n_out,W,activation)
        self.__init_b(n_out,b)
        self.__init_output(activation)
        self.params = [self.W, self.b]

    def __init_rng(self,theano_rng=None):
        self.rng = np.random.RandomState(123)
       
    def __init_W(self,n_in,n_out,W,activation):
        if W is None:
            bound=np.sqrt(6. / (n_in + n_out))
            init_value=self.rng.uniform(
              low=-bound,high=bound,size=(n_in, n_out)
            )
            W_values = np.asarray(
                init_value,
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)
        self.W = W

    def __init_b(self,n_out,b):
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        self.b = b

    def __init_output(self,activation):
        lin_output = T.dot(self.input_data, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

class MLP(object):
    def __init__(self, input_data, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(
            input_data=input_data,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )


        self.logRegressionLayer = logit.LogisticRegression(
            input_data=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )

        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )

        self.errors = self.logRegressionLayer.errors

        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        self.input_data = input_data

def build_model(dataset,learning_rate):
    n_hidden=500
    L1_reg=0.0 
    L2_reg=0.0
    symb_vars=logit.InputVariables()
    classifier = MLP(
        input_data=symb_vars.x,
        n_in=dataset.image_size,
        n_hidden=n_hidden,
        n_out=dataset.n_cats
    )

    cost = (
        classifier.negative_log_likelihood(symb_vars.y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    
    test_model = theano.function(
        inputs=[symb_vars.x,symb_vars.y],
        outputs=classifier.errors(symb_vars.y),
    )

    gparams = [T.grad(cost, param) for param in classifier.params]

    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    train_model = theano.function(
        inputs=[symb_vars.x,symb_vars.y],
        outputs=cost,
        updates=updates
    )
    return classifier,train_model


if __name__ == "__main__":
    path="/home/user/cf/conv_frames/cls/images/"
    dataset=load.get_images(path)
    cls=logit.learning_logit(dataset,build_model)
