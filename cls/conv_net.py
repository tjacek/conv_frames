import load,ml_tools,nn,learning
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

class ConvModel(object):
    def __init__(self,layers,pyx):
        self.layers=layers
        self.pyx=pyx

    def get_params(self):
        return [layer.w for layer in self.layers]

class ConvLayer(object):
    def __init__(self,layer,w):
        self.layer=layer
        self.w=w

def built_conv_cls(shape=(3200,20)):
    hyper_params=get_hyper_params()
    free_vars=ml_tools.SquareImages()
    model= create_conv_model(free_vars,hyper_params)
    train,test=create_conv_fun(free_vars,model,hyper_params)
    return ml_tools.Classifier(free_vars,model,train,test)

def create_conv_model(free_vars,hyper_params):
    kern_params=hyper_params['kern_params']
    l1=make_conv_layer(free_vars.X,kern_params[0],first=True)
    l2=make_conv_layer(l1.layer,kern_params[1])
    l3=make_conv_layer(l2.layer,kern_params[2],flat=True)
    l4,pyx=get_last_layer(l3.layer,(kern_params[3],kern_params[4]),0.0)
    layers=[l1,l2,l3,l4]
    return ConvModel(layers,pyx) 

def make_conv_layer(in_data,w_shape,p_drop_conv=0.0,
                    first=False,flat=False):
    w = ml_tools.init_kernels(w_shape)
    if(first):
        la = rectify(conv2d(in_data, w, border_mode='full'))
    else:
        la = rectify(conv2d(in_data, w))
    l = max_pool_2d(la, (2, 2))
    if(flat):
        l = T.flatten(l, outdim=2)
    layer=dropout(l, p_drop_conv) 
    return ConvLayer(layer,w)

def get_last_layer(l3,w_shape,p_drop_hidden=0.0):
    w = ml_tools.init_kernels(w_shape[0])
    l4 = rectify(T.dot(l3, w))
    l4 = dropout(l4, p_drop_hidden)
    w_0 = ml_tools.init_kernels(w_shape[1])
    pyx = softmax(T.dot(l4, w_0))
    layer_4=ConvLayer(l4,w)
    return layer_4,pyx

def create_conv_fun(free_vars,model,hyper_params):
    learning_rate=hyper_params['learning_rate']
    py_x=model.pyx
    loss=get_loss_function(free_vars,py_x)
    input_vars=free_vars.get_vars()
    params=model.get_params()
    update=RMSprop(loss, params, learning_rate)
    train = theano.function(inputs=input_vars, 
                                outputs=loss, updates=update, 
                                allow_input_downcast=True)
    y_pred = T.argmax(py_x, axis=1)
    test=theano.function(inputs=[free_vars.X], outputs=y_pred, 
            allow_input_downcast=True) 
    return train,test

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def get_loss_function(free_vars,py_x):
    return T.mean(T.nnet.categorical_crossentropy(py_x,free_vars.y))

def rectify(X):
    return T.maximum(X, 0.)

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def get_hyper_params(learning_rate=0.13):
    kern_params=[(32, 1, 3, 3),(64, 32, 3, 3),(128, 64, 3, 3),
                 (128 * 6 * 6, 625),(625, 7)]
    params={'learning_rate': learning_rate,
            'kern_params':kern_params}
    return params

if __name__ == '__main__':
    dataset_path="/home/user/cf/conv_frames/cls/images/"
    dataset=load.get_images(dataset_path)
    out_path="/home/user/cf/exp1/nn"
    cls=learning.create_classifer(dataset_path,out_path,built_conv_cls,flat=False)
    learning.evaluate_cls(dataset_path,out_path,flat=False)
