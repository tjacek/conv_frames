import load,ml_tools,nn,learning
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

class ConvModel(object):
    def __init__(self,conv_layers,last_layer):
        self.conv_layers=conv_layers
        self.last_layer=last_layer

    def get_params(self):
        conv_params=[layer.w for layer in self.conv_layers]
        ll_params=self.last_layer.get_params()
        return conv_params + ll_params
 
    def pyx(self):
        return self.last_layer.out_layer

class ConvLayer(object):
    def __init__(self,layer,w):
        self.layer=layer
        self.w=w

class LastLayer(object):
    def __init__(self,hidden_layer,w_h,
                      out_layer,w_o):
        self.hidden_layer=hidden_layer
        self.w_h=w_h
        self.out_layer=out_layer
        self.w_o=w_o

    def get_params(self):
        return [self.w_h,self.w_o]

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
    ll_shape=(kern_params[3],kern_params[4])
    last_layer=make_last_layer(l3.layer,ll_shape,0.0)
    layers=[l1,l2,l3]
    return ConvModel(layers,last_layer) 

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

def make_last_layer(l3,w_shape,p_drop_hidden=0.0):
    w_h = ml_tools.init_kernels(w_shape[0])
    hidden_layer = rectify(T.dot(l3, w_h))
    l4 = dropout(hidden_layer, p_drop_hidden)
    w_o = ml_tools.init_kernels(w_shape[1])
    out_layer = softmax(T.dot(l4, w_o))
    last_layer=LastLayer(hidden_layer,w_h,out_layer,w_o)
    return last_layer

def create_conv_fun(free_vars,model,hyper_params):
    learning_rate=hyper_params['learning_rate']
    py_x=model.pyx()
    loss=get_loss_function(free_vars,py_x)
    input_vars=free_vars.get_vars()
    params=model.get_params()
    update=ml_tools.sgd(loss, params, learning_rate)
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

def get_hyper_params(learning_rate=0.10):
    kern_one=8#32
    kern_two=12#64
    kern_third=16#128
    n_hidden=400
    kern_params=[(kern_one, 1, 3, 3),(kern_two, kern_one, 3, 3),(kern_third, kern_two, 3, 3),
                 (kern_third * 6 * 6, n_hidden),(n_hidden, 7)]
    params={'learning_rate': learning_rate,
            'kern_params':kern_params}
    return params

if __name__ == '__main__':
    dataset_path="/home/user/cf/conv_frames/cls/images/"
    dataset=load.get_images(dataset_path)
    out_path="/home/user/cf/exp1/conv_net"
    cls=learning.create_classifer(dataset_path,out_path,built_conv_cls,flat=False)
    learning.evaluate_cls(dataset_path,out_path,flat=False)
