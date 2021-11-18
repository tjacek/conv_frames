import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input, Dense,Conv2D,Reshape,Conv2DTranspose
from tensorflow.keras.layers import Flatten,MaxPooling2D,UpSampling2D
from keras import regularizers
from tensorflow.keras.models import load_model
#import tc_nn
from tensorflow.keras.utils import Sequence
import os.path
import data.imgs,files#data.seqs,learn,files

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

class FrameGenerator(Sequence):
    def __init__(self,frame_paths,batch_size,read):
        self.frame_paths=frame_paths
        self.batch_size=batch_size
        self.read=read

    def __len__(self):
        return int(len(self.frame_paths)/self.batch_size)

    def __getitem__(self, i):
        paths_i=self.frame_paths[i*self.batch_size:(i+1)*self.batch_size]
        x_i=np.array([self.read(path_j) for path_j in paths_i])
        return x_i,x_i

def make_frame_gen(in_path,batch_size=8,read=None):
    path_dict=files.get_path_dict(in_path)
    path_dict=path_dict.split()[0]
    frame_paths=[]
    for frame_i in path_dict.values():
        frame_paths+=frame_i
    return FrameGenerator(frame_paths,batch_size,read)

def train_ae(in_path,out_path,n_batch=8,n_epochs=30):
#    read=tc_nn.SimpleRead(dim=(64,128),preproc=data.imgs.Downsample())
#    seq_dict=read(in_path)
#    train=seq_dict.split()[0]
#    X,params=to_dataset(train,8)
#    model=get_train(params,out_path)
    params={ 'n_channels':3,"dims":(128,64)}
    read=data.imgs.ReadFrames(color="color")
    ae_gen=make_frame_gen(in_path,n_batch,read)
    if(os.path.isfile(out_path)):
        model=load_model(out_path)
    else:    
        make_ae=Autoencoder()
        model=make_ae(params)[0]
    model.fit(ae_gen,epochs=n_epochs)#,batch_size=n_batch)
    model.save(out_path)

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

#def to_dataset(train,fraction=2):
#    X=[]
#    for seq_i in train.values():
#        X+=seq_i    
#    if(fraction):
#        X=[x_j for j,x_j in enumerate( X)
#              if( (j%fraction)==0)]
#    X=np.array(X)
#    params={ 'n_channels':1,"dims":train.dims()}
#    return X,params

#def test_read(in_path,out_path):
#    read=tc_nn.SimpleRead(dim=(64,128),preproc=data.imgs.Downsample())
#    seq_dict=read(in_path)
#    seq_dict.save(out_path)

def ae_exp(frame_path,out_path,n_epochs=2):
    files.make_dir(out_path)
    model_path="%s/ae" % out_path
    seq_path="%s/seqs" % out_path 
#    train_ae(frame_path,model_path,n_epochs=n_epochs)
    extract_ae(frame_path,model_path,seq_path)

frame_path="../cc/florence"
train_ae(frame_path,"ae")
#ae_exp(frame_path,"ae",n_epochs=5)
#reconstruct(frame_path,"%s/ae" % out_path,"recon")