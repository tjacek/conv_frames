import load
import numpy as np
import theano
import theano.tensor as T
import imp
utils =imp.load_source("utils","/home/user/cf/conv_frames/utils.py")

class LogisticRegression(object):

    def __init__(self, input_data, n_in, n_out):
        self.init_weights(n_in,n_out)
        self.init_bias(n_out)

        self.p_y_given_x = T.nnet.softmax(T.dot(input_data, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.input_data = input_data

    def init_weights(self,n_in,n_out):
        init_value=np.zeros((n_in, n_out),dtype=theano.config.floatX)
        self.W = theano.shared(value=init_value,name='W',borrow=True)

    def init_bias(self,n_out):
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

def build_model(learning_rate):
    symb_vars=InputVariables()
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

class InputVariables(object):
    def __init__(self):
        self.x = T.matrix('x')
        self.y = T.lvector('y')

class TrainParams(object):
    def __init__(self,n_batches):
        self.patience = 5000  
        self.patience_increase = 2  
        self.improvement_threshold = 0.995  
        self.validation_frequency = min(n_batches,self.patience / 2)

def learning_logit(dataset,n_epochs=1000,learning_rate=0.13,batch_size=5):

    X_b,y_b=dataset.get_batches(batch_size)
    n_train_batches=len(y_b)
    # build model
    classifier,train_model=build_model(learning_rate)
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

if __name__ == "__main__":
    path="/home/user/cf/conv_frames/cls/images/"
    dataset=load.get_images(path)
    cls=learning_logit(dataset)
