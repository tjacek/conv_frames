import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input, Dense,Conv2D,Reshape,Conv2DTranspose
from tensorflow.keras.layers import Flatten,MaxPooling2D,UpSampling2D
from keras import regularizers
import tc_nn
import data.imgs

class Autoencoder(object):
    def __init__(self,n_hidden=100):
        self.n_hidden=n_hidden
        self.n_kerns=64
        self.scale=(2,2)
    
    def __call__(self,params):
        x,y=params["dims"]
        input_img = Input(shape=(x,y, params['n_channels']))
        x = Conv2D(self.n_kerns, (5, 5), activation='relu',padding='same')(input_img)
        x = MaxPooling2D(self.scale)(x)
        x = Conv2D(16, (5, 5), activation='relu',padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        shape = K.int_shape(x)
        x=Flatten()(x)
        encoded=Dense(100,name='hidden',kernel_regularizer=regularizers.l1(0.01))(x)    
        x = Dense(shape[1]*shape[2]*shape[3])(encoded)
        x = Reshape((shape[1], shape[2], shape[3]))(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2DTranspose(16, (5, 5), activation='relu',padding='same')(x)
        x = UpSampling2D(self.scale)(x)
        x = Conv2DTranspose(self.n_kerns, (5, 5), activation='relu',padding='same')(x)
    
        x=Conv2DTranspose(filters=params['n_channels'],kernel_size=self.n_kerns,padding='same')(x)
        recon=Model(input_img,encoded)
        autoencoder = Model(input_img, x)

        autoencoder.compile(optimizer='adam',
                      loss='mean_squared_error')#CustomLoss(autoencoder)
        autoencoder.summary()
        return autoencoder,recon

def train_ae(in_path):
    read=tc_nn.SimpleRead(dim=(64,64),preproc=data.imgs.Downsample())
    seq_dict=read(in_path)
    train=seq_dict.split()[0]
    X=[]
    for seq_i in train.values():
        X+=seq_i	
    X=np.array(X)
    params={ 'n_channels':1,"dims":train.dims()}
    make_ae=Autoencoder()
    model=make_ae(params)[0]
    model.fit(X,X,epochs=5,batch_size=16)

in_path="../best2/frames"
train_ae(in_path)