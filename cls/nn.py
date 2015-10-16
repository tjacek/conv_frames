import load,logit
import numpy as np
import theano
import theano.tensor as T
import ml_tools 

class MlpModel(object):
    def __init__(self,hidden,logistic):
        self.hidden=hidden
        self.logistic=logistic
        
    def get_params(self):
        return self.hidden + self.logistic

def create_mlp_model(params):
    n_in=params['n_in']
    n_hidden=params['n_hidden']
    n_out=params['n_out']
    hidden=logit.create_layer()
    logistic=logit.create_layer()
    return MlpModel(hidden,logit)

def build_model(dataset,model_params):
    learning_rate=model_params['learning_rate']
    n_hidden=model_params['n_hidden']
    L1_reg=model_params['L1_reg']
    L2_reg=model_params['L2_reg']
    symb_vars=ml_tools.InputVariables()
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

def get_default_params(learning_rate=0.13):
    params={'learning_rate': learning_rate,
            'n_hidden':500,'L1_reg':0.0,'L2_reg':0.0}
    return params

if __name__ == "__main__":
    path="/home/user/cf/conv_frames/cls/images/"
    dataset=load.get_images(path)
    params=get_default_params()
    cls=ml_tools.learning_iter(dataset,build_model,params)
