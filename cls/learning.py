import imp
utils =imp.load_source("utils","/home/user/cf/conv_frames/utils.py")
import theano
import theano.tensor as T
import numpy as np
import load

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

def as_array(X):
    return np.asarray(X, dtype=theano.config.floatX)

def evaluate_cls(dataset_path,cls_path,flat=True):
    dataset=load.get_images(dataset_path)
    cls=utils.read_object(cls_path) #learning_iter(dataset,cls)
    correct=check_prediction(dataset,cls,flat)
    print(correct)

def create_classifer(in_path,out_path,built_classifer,flat=True):
    dataset=load.get_images(in_path)
    cls=built_classifer(dataset.shape())
    print(flat)
    if(flat):
        cls=learning_iter(dataset,cls)
    else:
        cls=learning_conv(dataset,cls)
    utils.save_object(out_path,cls)
    return cls

def learning_iter(dataset,cls,
                  n_epochs=400,batch_size=40):
    X_b,y_b=dataset.single_batch(True)
    n_train_batches=len(y_b)
    # build model
    print '... training the model'
    #train_params=TrainParams()
    timer = utils.Timer()
    for epoch in xrange(n_epochs):
        c = []
        for batch_index in xrange(n_train_batches):
            x_i=X_b[batch_index]
            y_i=y_b[batch_index]
            c.append(cls.train(x_i,y_i))

        print 'Training epoch %d, cost ' % epoch, np.mean(c)

    timer.stop()
    print("Training time %d ",timer.total_time)
    return cls

def learning_conv(dataset,cls,
                  n_epochs=1000,batch_size=40):
    X_b,y_b=dataset.square_images()
    n_train_batches=len(y_b)
    # build model
    print '... training the model'
    timer = utils.Timer()
    for epoch in xrange(n_epochs):
        c = []
        c.append(cls.train(X_b,y_b))
        print 'Training epoch %d, cost ' % epoch, np.mean(c)

    timer.stop()
    print("Training time %d ",timer.total_time)
    return cls

def check_prediction(dataset,cls,flat=True):
    if(flat):
        X_b,y_b=dataset.get_batches(1)
        y=[cls.test(x_i)[0] for x_i in X_b]
    else:
        X_b,y_b=dataset.square_images()
        y=cls.test(X_b)
    print(X_b.shape)

    print(y)
    print(dataset.y)
    pred=(y==dataset.y)
    pred.astype(int)
    return np.mean(pred)

if __name__ == "__main__":
    dataset_path="/home/user/cf/conv_frames/cls/images/"
    dataset=load.get_images(dataset_path)
    cls_path="/home/user/cf/exp1/conv"
    evaluate_cls(dataset_path,cls_path)
        
