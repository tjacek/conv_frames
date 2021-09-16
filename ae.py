import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input, Dense,Conv2D,Reshape,Conv2DTranspose
from tensorflow.keras.layers import Flatten,MaxPooling2D,UpSampling2D
from keras import regularizers
from tensorflow.keras.models import load_model
import tc_nn
import os.path
import data.imgs,data.seqs,learn,files

class Autoencoder(object):
    def __init__(self,n_hidden=128):
        self.n_hidden=n_hidden
        self.n_kerns=[64,64,64]
        self.scale=[(4,4),(2,2),(2,2)]
    
    def __call__(self,params):
        x,y=params["dims"]
        n_layers=len(self.n_kerns)
        input_img = Input(shape=(x,y, params['n_channels']))
        x=input_img   
        for i in range(n_layers):
            x = Conv2D(self.n_kerns[i], (5, 5), activation='relu',padding='same')(x)
            x = MaxPooling2D(self.scale[i])(x)
        shape = K.int_shape(x)
        x=Flatten()(x)
        reg=None#regularizers.l1(0.01)
        encoded=Dense(self.n_hidden,name='hidden',kernel_regularizer=reg)(x)    
        x = Dense(shape[1]*shape[2]*shape[3])(encoded)
        x = Reshape((shape[1], shape[2], shape[3]))(x)
        for i in range(n_layers):
            j=n_layers-i-1
            x = UpSampling2D(self.scale[j])(x)
            x = Conv2DTranspose(self.n_kerns[i], (5, 5), activation='relu',padding='same')(x) 
        x=Conv2DTranspose(filters=params['n_channels'],kernel_size=self.n_kerns[0],padding='same')(x)
        recon=Model(input_img,encoded)
        autoencoder = Model(input_img, x)

        autoencoder.compile(optimizer='adam',
                      loss='mean_squared_error')#CustomLoss(autoencoder)
        autoencoder.summary()
        return autoencoder,recon

def train_ae(in_path,out_path,n_batch=8,n_epochs=30):
    read=tc_nn.SimpleRead(dim=(64,128),preproc=data.imgs.Downsample())
    seq_dict=read(in_path)
    train=seq_dict.split()[0]
    X,params=to_dataset(train,8)
    model=get_train(params,out_path)
    model.fit(X,X,epochs=n_epochs,batch_size=n_batch)
    model.save(out_path)

def get_train(params,out_path):
    if(os.path.isfile(out_path)):
        model=load_model(out_path)
    else:    
        make_ae=Autoencoder()
        model=make_ae(params)[0]
    return model

def extract_ae(in_path,nn_path,out_path):
    read=tc_nn.SimpleRead(dim=(64,128),preproc=data.imgs.Downsample())
    extract=learn.ExtractSeqs(read)
    extract(in_path,nn_path,out_path)

def reconstruct(in_path,nn_path,out_path):
    read=tc_nn.SimpleRead(dim=(64,128),preproc=data.imgs.Downsample())
    frame_dict=read(in_path)
    model=learn.base_read_model(frame_dict,nn_path)
    def helper(img_i):
        img_i=np.array(img_i)#,axis=-1)
        return model.predict(img_i)
    frame_dict=frame_dict.transform(helper,new=True,single=False)
    frame_dict.save(out_path)

def to_dataset(train,fraction=2):
    X=[]
    for seq_i in train.values():
        X+=seq_i    
    if(fraction):
        X=[x_j for j,x_j in enumerate( X)
              if( (j%fraction)==0)]
    X=np.array(X)
    params={ 'n_channels':1,"dims":train.dims()}
    return X,params

def test_read(in_path,out_path):
    read=tc_nn.SimpleRead(dim=(64,128),preproc=data.imgs.Downsample())
    seq_dict=read(in_path)
    seq_dict.save(out_path)

def ae_exp(frame_path,out_path,n_epochs=2):
    files.make_dir(out_path)
    model_path="%s/ae" % out_path
    seq_path="%s/seqs" % out_path 
#    train_ae(frame_path,model_path,n_epochs=n_epochs)
    extract_ae(frame_path,model_path,seq_path)

frame_path="../best2/frames"
out_path="../best2/3_layers"
ae_exp(frame_path,out_path,n_epochs=5)
#reconstruct(frame_path,"%s/ae" % out_path,"recon")