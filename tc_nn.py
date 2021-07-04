import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D
from tensorflow.keras.models import Sequential
from tcn import TCN, tcn_full_summary
from keras.models import load_model
import tensorflow.keras.losses
import data.imgs,learn,files,ens

class TC_NN(object):
    def __init__(self,n_hidden=100):
        self.n_kern=[64,64,64]
        self.kern_size=[(5,5),(5,5),(5,5)]
        self.pool_size=[(2,2),(2,2),(2,2)]
        self.n_hidden=n_hidden

    def __call__(self,params):
        inputs = Input(shape=(params['seq_len'],*params['dims'],1))
        x = Lambda(lambda y: K.reshape(y, (-1, *params['dims'], 1)))(inputs)
        for i,n_kern_i in enumerate(self.n_kern):
            x=Conv2D(n_kern_i, kernel_size=self.kern_size[i])(x)
            x=MaxPool2D(pool_size=self.pool_size[i])(x)
        num_features_cnn = np.prod(K.int_shape(x)[1:])
        x = Lambda(lambda y: K.reshape(y, (-1, params['seq_len'], num_features_cnn)))(x)
        x = TCN(self.n_hidden,name="hidden",use_batch_norm=True)(x)
        x = Dense(params['n_cats'], activation='sigmoid')(x)
        model = Model(inputs=[inputs], outputs=[x])
        tcn_full_summary(model, expand_residual_blocks=False)
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
        return model

def ensemble_exp(frame_path,ens_path,n_epochs=5):
    train,extract,read=make_single_exp()
    ensemble=ens.BinaryEns(read,train,extract)
    ensemble(frame_path,ens_path,n_epochs)

def single_exp(in_path,out_path,n_epochs=100):
    paths=files.prepare_dirs(out_path,["nn","feats"])
    print(paths)
    train,extract,read=make_single_exp()
    train(in_path,paths["nn"],n_epochs=n_epochs)
    extract(in_path,paths["nn"],paths["feats"])

def make_single_exp():
    read=get_read(seq_len=20,dim=(64,64))
    make_nn=TC_NN()
    train=learn.Train(to_dataset,make_nn,read=read)
    extract=learn.Extract(make_nn,read)
    return train,extract,read

def get_read(seq_len=20,dim=(64,64)):
    def helper(in_path):
        frame_seq=data.imgs.read_frame_seqs(in_path,n_split=1)
        frame_seq.subsample(seq_len)
        frame_seq.scale(dim)
        return frame_seq
    return helper

def to_dataset(frames):
    X,y=frames.to_dataset()
    params={'seq_len':X.shape[1],'dims':(X.shape[2],X.shape[3]),
                'n_cats':frames.n_cats()}
    return X,y,params

in_path="../MSR/frames"
ensemble_exp(in_path,"ens",n_epochs=5)