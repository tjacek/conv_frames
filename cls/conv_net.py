import load,ml_tools,nn,logit
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

class LeNetConvPoolLayer(object):

    def __init__(self, input_data, filter_shape, image_shape, poolsize=(2, 2)):

        assert image_shape[1] == filter_shape[1]
        self.input_data = input_data
        self.rng=ml_tools.RandomNum()
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))

        self.__init_W(fan_in,fan_out,filter_shape)
        self.__init_b(filter_shape)
        self.__init_output(filter_shape,image_shape,poolsize)
        self.params = [self.W, self.b]

    def __init_W(self,fan_in,fan_out,filter_shape):
        init_value=self.rng.uniform(fan_in,fan_out,filter_shape)
        self.W = theano.shared(
            np.asarray(
                init_value,
                dtype=theano.config.floatX
            ),
            borrow=True
        )

    def __init_b(self,filter_shape):
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

    def __init_output(self,filter_shape,image_shape,poolsize):
        conv_out = conv.conv2d(
            input=self.input_data,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

def build_model(dataset,model_params):
    batch_size=model_params['batch_size']
    learning_rate=model_params['learning_rate']
    nkerns=model_params['nkerns']

    symb_vars=ml_tools.InputVariables()
    image_dim=dataset.image_shape
    input0_shape=(batch_size,1,image_dim[0],image_dim[1])
    layer0_input = symb_vars.x.reshape(input0_shape)

    layer0 = LeNetConvPoolLayer(
        input_data=layer0_input,
        image_shape=input0_shape,
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    input1_shape=(batch_size,nkerns[0],38, 18)
    layer1 = LeNetConvPoolLayer(
        input_data=layer0.output,
        image_shape=input1_shape,
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    layer2_input = layer1.output.flatten(2)
    print(nkerns)
    layer2 = nn.HiddenLayer(
        input_data=layer2_input,
        n_in=nkerns[1] * (59*2 +1),
        n_out=500,
        activation=T.tanh
    )

    layer3 = logit.LogisticRegression(input_data=layer2.output, 
                           n_in=500, n_out=dataset.n_cats)

    cost = layer3.negative_log_likelihood(symb_vars.y)

    test_model = theano.function(
        [symb_vars.x, symb_vars.y],
        layer3.errors(symb_vars.y),
    )

    params = layer3.params + layer2.params + layer1.params + layer0.params 
    grads = T.grad(cost, params)

    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [symb_vars.x, symb_vars.y],
        cost,
        updates=updates
    )
    return layer3,train_model

def get_default_params(learning_rate=0.13):
    params={'learning_rate': learning_rate,
            'nkerns':[20, 50],
            'batch_size':1}
    return params

if __name__ == '__main__':
    path="/home/user/cf/conv_frames/cls/images/"
    dataset=load.get_images(path)
    params=get_default_params()
    cls=ml_tools.learning_iter(dataset,build_model,params,n_epochs=100)
