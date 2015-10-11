import load
import numpy as np
import theano
import theano.tensor as T
import ml_tools

class FlatImages(object):
    def __init__(self):
        self.X=T.matrix('x')
        self.y=T.lvector('y')

    def get_vars(self):
        return [self.X,self.y]

class LogitModel(object):
    def __init__(self,W,b):
        self.W=W
        self.b=b

    def get_params(self):
        return [self.W,self.b]

class LogitClassifier(object):
    def __init__(self,free_vars,model,train,test):
        self.free_vars=free_vars
        self.model=model
        self.test=test
        self.train=train

def built_classifer(shape=(3200,20)):
    free_vars=FlatImages()
    model=create_model(shape)
    train,test=create_cls_fun(free_vars,model)
    return LogitClassifier(free_vars,model,train,test)

def create_model(shape):
    W=init_weights(shape[0],shape[1])
    b=init_bias(shape[1])
    return LogitModel(W,b)

def init_weights(n_in,n_out):
    init_value=ml_tools.get_zeros((n_in,n_out))
    return theano.shared(value=init_value,name='W',borrow=True)

def init_bias(n_out):
    init_value=ml_tools.get_zeros((n_out,))
    return theano.shared(value=init_value,name='b',borrow=True)

def create_cls_fun(free_vars,model):
    learning_rate=0.15
    py_x=get_px_y(free_vars,model)
    loss=get_loss_function(free_vars,py_x)
    input_vars=free_vars.get_vars()
    g_W = T.grad(cost=loss, wrt=model.W)
    g_b = T.grad(cost=loss, wrt=model.b)
    update = [(model.W, model.W - learning_rate * g_W),
               (model.b,model.b - learning_rate * g_b)]
    train = theano.function(inputs=input_vars, 
                                outputs=loss, updates=update, 
                                allow_input_downcast=True)
    y_pred = T.argmax(py_x, axis=1)
    test=theano.function(inputs=[free_vars.X], outputs=y_pred, 
            allow_input_downcast=True) 
    return train,test

def get_px_y(free_vars,model):    
    equation=T.dot(free_vars.X, model.W) + model.b
    return T.nnet.softmax(equation)

def get_loss_function(free_vars,py_x):
    return T.mean(T.nnet.categorical_crossentropy(py_x,free_vars.y))

if __name__ == "__main__":
    dataset_path="/home/user/cf/conv_frames/cls/images/"
    cls=built_classifer()
    out_path="/home/user/cf/exp1/logit"
    #ml_tools.evaluate_cls(dataset_path,cls)
    ml_tools.create_classifer(dataset_path,out_path,cls)
