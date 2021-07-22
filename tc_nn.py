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

class ReadFrames(object):
    def __init__(self,seq_len=20,dim=(64,64),n_split=1,agum=1):
        self.seq_len=seq_len
        self.dim=dim
        self.n_split=n_split
        self.agum=agum

    def __call__(self,in_path):
        frame_seq=data.imgs.read_frame_seqs(in_path,n_split=self.n_split)
        if(self.agum is None or self.agum==1):
            frame_seq.subsample(self.seq_len)
        else:
            sample=data.imgs.MinLength(self.seq_len)
            def agum_fun(frames):
                return [ sample(frames) for i in range(self.agum)]
            frame_seq=frame_seq.agum(agum_fun)
        frame_seq.scale(self.dim)
        return frame_seq


class TC_NN(object):
    def __init__(self,n_hidden=100,loss='binary_crossentropy',batch=True,
        n_kern=[64,64,64]):
        self.n_kern=n_kern#[64,64,64]
        self.kern_size=[(5,5),(5,5),(5,5)]
        self.pool_size=[(2,2),(2,2),(2,2)]
        self.n_hidden=n_hidden
        self.loss=loss
        self.batch=batch

    def __call__(self,params):
        inputs = Input(shape=(params['seq_len'],*params['dims'],1))
        x = Lambda(lambda y: K.reshape(y, (-1, *params['dims'], 1)))(inputs)
        for i,n_kern_i in enumerate(self.n_kern):
            x=Conv2D(n_kern_i, kernel_size=self.kern_size[i])(x)
            x=MaxPool2D(pool_size=self.pool_size[i])(x)
        num_features_cnn = np.prod(K.int_shape(x)[1:])
        x = Lambda(lambda y: K.reshape(y, (-1, params['seq_len'], num_features_cnn)))(x)
        x = TCN(self.n_hidden,name="hidden",use_batch_norm=self.batch)(x)
        x = Dense(params['n_cats'], activation='sigmoid')(x)
        model = Model(inputs=[inputs], outputs=[x])
        tcn_full_summary(model, expand_residual_blocks=False)
        model.compile('adam',self.loss, metrics=['accuracy'])
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

def to_dataset(frames):
    X,y=frames.to_dataset()
    params={'seq_len':X.shape[1],'dims':(X.shape[2],X.shape[3]),
                'n_cats':frames.n_cats()}
    return X,y,params

def read_model(frame_seq,nn_path,make_nn,params=None):
    if(params is None):
        params={'seq_len':frame_seq.min_len(),'dims':frame_seq.dims(),
                'n_cats':frame_seq.n_cats()}
    model=make_nn(params)
    model.load_weights(nn_path)
    return model

def save(model,out_path):
    if(out_path):
        model.save_weights(out_path)

if __name__ == "__main__":
    in_path="../MSR/frames"
    ensemble_exp(in_path,"ens",n_epochs=5)