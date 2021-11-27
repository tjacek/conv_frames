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
import data.imgs,files,learn#data.seqs,files

class Autoencoder(object):
    def __init__(self,n_hidden=256):
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

class LazyGenerator(Sequence):
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

class FrameGenerator(Sequence):
    def __init__(self,frames,batch_size):
        self.frames=frames
        self.batch_size=batch_size

    def __len__(self):
        return int(len(self.frames)/self.batch_size)

    def __getitem__(self, i):
        x_i=self.frames[i*self.batch_size:(i+1)*self.batch_size]
        x_i=np.array(x_i)
        return x_i,x_i

def make_lazy_gen(in_path,batch_size=8,read=None):
    path_dict=files.get_path_dict(in_path)
    path_dict=path_dict.split()[0]
    frame_paths=[]
    for frame_i in path_dict.values():
        frame_paths+=frame_i
    return LazyGenerator(frame_paths,batch_size,read)

def make_frame_gen(in_path,batch_size=8,read=None):
    frame_seqs=data.imgs.read_frame_seqs(in_path,read)
    frame_seqs=frame_seqs.split()[0]
    X=[]
    for name_i,seq_i in frame_seqs.items():
        X+=seq_i
#    raise Exception(len(X))
    return FrameGenerator(X,batch_size)

def train_ae(in_path,out_path,n_batch=8,n_epochs=30):
    params={ 'n_channels':3,"dims":(128,64)}
    read=data.imgs.ReadFrames(color="color")
    ae_gen=make_frame_gen(in_path,n_batch,read)
    if(os.path.exists(out_path)):
        model=load_model(out_path)
    else:    
        make_ae=Autoencoder()
        model=make_ae(params)[0]
    model.fit(ae_gen,epochs=n_epochs)
    model.save(out_path)

def extract_ae(in_path,nn_path,out_path):
    read=tc_nn.SimpleRead(dim=(64,128),preproc=data.imgs.Downsample())
    extract=learn.ExtractSeqs(read)
    extract(in_path,nn_path,out_path)

def reconstruct(in_path,nn_path,out_path,diff=False):
    read=data.imgs.ReadFrames(color="color")
    model=tf.keras.models.load_model(nn_path)
    if(diff):
        def helper(name_i,frames):
            print(name_i)
            new_frames=model.predict(np.array(frames))
            return [ np.abs(frame_i-new_frame_i)
                for frame_i,new_frame_i in zip(new_frames,frames)]
    else:
        def helper(name_i,frames):
            frames=np.array(frames)
            return model.predict(frames)
    data.imgs.transform_lazy(in_path,out_path,helper,read,
            recreate=True,single=False)

def ae_exp(frame_path,out_path,n_epochs=5):
    files.make_dir(out_path)
    model_path="%s/ae" % out_path
    seq_path="%s/seqs" % out_path 
#    train_ae(frame_path,model_path,n_epochs=n_epochs)
    extract_ae(frame_path,model_path,seq_path)

frame_path="../cc2/final"
#train_ae(frame_path,"../cc2/ae",n_epochs=100)
#ae_exp(frame_path,"ae",n_epochs=5)
reconstruct(frame_path,"../cc2/ae","../cc2/recon",diff=False)