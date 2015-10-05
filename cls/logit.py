import load
import numpy as np
import theano
import theano.tensor as T
import ml_tools

class LogisticRegression(object):

    def __init__(self, input_data, n_in, n_out):
        self.__init_weights(n_in,n_out)
        self.__init_bias(n_out)

        self.p_y_given_x = T.nnet.softmax(T.dot(input_data, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.input_data = input_data

    def __init_weights(self,n_in,n_out):
        init_value=np.zeros((n_in, n_out),dtype=theano.config.floatX)
        self.W = theano.shared(value=init_value,name='W',borrow=True)

    def __init_bias(self,n_out):
        init_value=np.zeros((n_out,),dtype=theano.config.floatX)
        self.b = theano.shared(value=init_value,name='b',borrow=True)

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def build_model(dataset,model_params):
    learning_rate=model_params['learning_rate']
    symb_vars=ml_tools.InputVariables()
    classifier = LogisticRegression(input_data=symb_vars.x, 
                  n_in=dataset.image_size, n_out=dataset.n_cats)
    cost = classifier.negative_log_likelihood(symb_vars.y)

    test_model = theano.function(
        inputs=[symb_vars.x,symb_vars.y],
        outputs=classifier.errors(symb_vars.y))

    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    train_model = theano.function(
        inputs=[symb_vars.x,symb_vars.y],
        outputs=cost,
        updates=updates)
    return classifier,train_model

def get_default_params(learning_rate=0.13):
    return {'learning_rate': learning_rate}

if __name__ == "__main__":
    path="/home/user/cf/conv_frames/cls/images/"
    dataset=load.get_images(path)
    params=get_default_params()
    cls=ml_tools.learning_iter(dataset,build_model,params)
