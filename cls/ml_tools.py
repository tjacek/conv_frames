import imp
utils =imp.load_source("utils","/home/user/cf/conv_frames/utils.py")
import theano
import theano.tensor as T
import numpy as np

class InputVariables(object):
    def __init__(self):
        self.x = T.matrix('x')
        self.y = T.lvector('y')

class RandomNum(object):
    def __init__(self):
        self.rng = np.random.RandomState(123)

    def uniform(self,n_in,n_out,dim):
        bound=np.sqrt(6. / (n_in + n_out))
        #dim=(n_in, n_out)
        return self.rng.uniform(-bound,bound,dim)

class TrainParams(object):
    def __init__(self,n_batches):
        self.patience = 5000  
        self.patience_increase = 2  
        self.improvement_threshold = 0.995  
        self.validation_frequency = min(n_batches,self.patience / 2)

def learning_iter(dataset,make_model,model_params,n_epochs=1000,batch_size=5):

    X_b,y_b=dataset.get_batches(batch_size)
    n_train_batches=len(y_b)
    # build model
    classifier,train_model=make_model(dataset,model_params)
    print '... training the model'
    #train_params=TrainParams()

    timer = utils.Timer()
    for epoch in xrange(n_epochs):
        c = []
        for batch_index in xrange(n_train_batches):
            x_i=X_b[batch_index]
            y_i=y_b[batch_index]
            c.append(train_model(x_i,y_i))

        print 'Training epoch %d, cost ' % epoch, np.mean(c)

    timer.stop()
    print("Training time %d ",timer.total_time)
    return classifier
