import load,logit
import numpy as np
import theano
import theano.tensor as T
import ml_tools,learning

class MlpModel(object):
    def __init__(self,hidden,logistic):
        self.hidden=hidden
        self.logistic=logistic
        
    def get_params(self):
        return self.hidden.get_params() + self.logistic.get_params()

def built_nn_cls(shape=(3200,20)):
    hyper_params=get_hyper_params()
    free_vars=ml_tools.FlatImages()
    model= create_mlp_model(hyper_params)
    train,test=create_nn_fun(free_vars,model,hyper_params)
    return ml_tools.Classifier(free_vars,model,train,test)

def create_mlp_model(hyper_params):
    n_in=hyper_params['n_in']
    n_hidden=hyper_params['n_hidden']
    n_out=hyper_params['n_out']
    hidden=ml_tools.create_layer((n_in,n_hidden))
    logistic=ml_tools.create_layer((n_hidden,n_out))
    return MlpModel(hidden,logistic)

def create_nn_fun(free_vars,model,hyper_params):
    learning_rate=hyper_params['learning_rate']
    py_x=get_px_y(free_vars,model)
    loss=get_loss_function(free_vars,py_x)
    input_vars=free_vars.get_vars()
    params=model.get_params()
    update=sgd(loss, params, learning_rate)
    train = theano.function(inputs=input_vars, 
                                outputs=loss, updates=update, 
                                allow_input_downcast=True)
    y_pred = T.argmax(py_x, axis=1)
    test=theano.function(inputs=[free_vars.X], outputs=y_pred, 
            allow_input_downcast=True) 
    return train,test

def get_px_y(free_vars,model):
    hidden=model.hidden
    output_layer=model.logistic
    h = T.nnet.sigmoid(T.dot(free_vars.X, hidden.W) + hidden.b)
    pyx = T.nnet.softmax(T.dot(h, output_layer.W) + output_layer.b)
    return pyx

def get_loss_function(free_vars,py_x):
    return T.mean(T.nnet.categorical_crossentropy(py_x,free_vars.y))

def sgd(loss, params, learning_rate=0.05):
    gparams = [T.grad(loss, param) for param in params]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(params, gparams)
    ]
    return updates

def get_hyper_params(learning_rate=0.15):
    params={'learning_rate': learning_rate,
            'n_in':3200,'n_out':7,'n_hidden':500,}
    return params

if __name__ == "__main__":
    dataset_path="/home/user/cf/conv_frames/cls/images/"
    dataset=load.get_images(dataset_path)
    out_path="/home/user/cf/exp1/nn"
    learning.create_classifer(dataset_path,out_path,built_nn_cls)
