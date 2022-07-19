#import tensorflow as tf
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#print("physical_devices-------------", len(physical_devices))
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np
from tensorflow import keras 
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.wrappers import TimeDistributed
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.recurrent import LSTM
from keras.layers.normalization import layer_normalization
import tensorflow.keras.optimizers
from keras import regularizers
import cv2
import os.path
import gen,deep,data.feats,learn,files,agum

class FRAME_LSTM(object):
    def __init__(self,dropout=None,activ='relu',batch=False,l1=None,optim_alg=None):
        if(optim_alg is None):
            optim_alg=tensorflow.keras.optimizers.Adam(learning_rate=0.00001)
        self.dropout=dropout
        self.activ=activ
        self.batch=batch
        self.l1=l1
        self.optim_alg=optim_alg #"adam"

    def __call__(self,params):
        input_shape= (params['seq_len'],*params['dims']) 
        model=Sequential()
        n_kern,kern_size,pool_size=[96,64,64],[(5,5),(5,5),(5,5)],[(2,2),(2,2),(2,2)]
        lstm_cnn(model,n_kern,kern_size,pool_size,self.activ,input_shape)
        model.add(TimeDistributed(Flatten()))
        n_hidden= [256,128,64] #[384,192,96]
        model.add(TimeDistributed(Dense(n_hidden[0])))
        if( not (self.dropout is None)):
            model.add(TimeDistributed(Dropout(self.dropout)))	
        reg=None if(self.l1  is None) else regularizers.l1(self.l1)
        model.add(TimeDistributed(Dense(n_hidden[1], name="first_dense",kernel_regularizer=reg)))
        
        model.add(LSTM(n_hidden[2], return_sequences=True, name="lstm_layer"));
		
        if(self.batch):
            model.add(GlobalAveragePooling1D(name="prebatch"))
            model.add(BatchNormalization(name="global_avg"))
        else:
            model.add(GlobalAveragePooling1D(name="global_avg"))
        model.add(Dense(params['n_cats'],activation='softmax'))

        model.compile(loss='categorical_crossentropy',
            optimizer=self.optim_alg,#keras.optimizers.Adadelta(),
            metrics=['accuracy'])
        model.summary()
        return model

def lstm_cnn(model,n_kern,kern_size,pool_size,activ,input_shape):
    for i,n_kern_i in enumerate(n_kern):
        if(i==0):
            conv_i=Conv2D(filters= n_kern_i,kernel_size=kern_size[i], padding='same')
            model.add(TimeDistributed(conv_i,input_shape=input_shape))
        else:
            conv_i=Conv2D(n_kern_i,kern_size[i])
            model.add(TimeDistributed(conv_i))
        model.add(TimeDistributed(Activation(activ)))
        model.add(TimeDistributed(MaxPooling2D(pool_size=pool_size[i])))

def ens(in_path,out_path,n_cats=12,n_epochs=25,
        modality='grey',dims=(96, 128,1)):
    files.make_dir(out_path)
    files.make_dir("%s/nn" % out_path)
    files.make_dir("%s/feats" % out_path)
    agum_fun=[[],agum.flip]
    read=data.imgs.ReadFrames(color=modality)
    subsample=data.imgs.MinLength(30)
#    subsample=data.imgs.StaticDownsample(30)
    params={'seq_len':30,'dims':dims,"n_cats":2,
            "read":read,"agum":agum_fun,"subsample":subsample}
    n_frames,n_batch=None,8
    batch_gen=gen.make_batch_gen(in_path,
        n_frames,n_batch,read=params["read"],subsample=params["subsample"])
    if(type(n_cats)==int):
        clfs=range(n_cats)
    if(type(n_cats)==tuple):
        clfs=range(*n_cats)
    for i in clfs:
        gen_i=gen.BinaryGenerator(i,batch_gen)
        gen_i=agum.add_agum(gen_i,params["agum"])
        nn_i="%s/nn/%d" % (out_path,i)
        train(gen_i,nn_i,params,n_epochs=n_epochs)
        feat_i="%s/feats/%d" % (out_path,i)
        extract(in_path,nn_i,feat_i,params)#size=30)

def train(generator,nn_path,params,n_epochs=20):
    if(type(generator)==str):
        n_frames,n_batch=512,8
        read=data.imgs.ReadFrames(color=cv2.IMREAD_COLOR)
        sampler=gen.make_lazy_sampler(in_path,read=read)
        batch_gen=gen.BatchGenerator(sampler,n_frames,n_batch)
        generator=gen.AllGenerator(batch_gen)
    if(not os.path.exists(nn_path)):
        make_model=FRAME_LSTM()
        model=make_model(params)
    else:
        model=learn.base_read_model(None,nn_path)
    model.fit(generator,epochs=n_epochs)
    model.save(nn_path)

def extract(in_path,nn_path,out_path,params):#size=30):
#    read=data.imgs.ReadFrames(color="color")
#    subsample=data.imgs.MinLength(size)# StaticDownsample(size)
    model=learn.base_read_model(None,nn_path)
    extractor=learn.get_extractor(model,"global_avg")
    read,subsample,agum_fun=params["read"],params["subsample"],params["agum"]
    def get_seq(in_path):
        frames=read(in_path)
        frames=subsample(frames)
        frames=np.array(frames)
        return frames
    def helper(seq_i):    
        frames=np.expand_dims(seq_i,0)
        return extractor.predict(frames)
#    agum=[[]]#,gen.flip]#,gen.reverse,[gen.flip,gen.reverse]]
    agum_extr=agum.AgumExtractor(get_seq,helper,agum_fun)
    feat_seq=data.feats.get_feats(in_path,agum_extr)
    feat_seq.save(out_path)

def single_exp(in_path,out_path,n_epochs=20):
    agum=[[],gen.flip]
    read=data.imgs.ReadFrames(color="color")
    subsample=data.imgs.MinLength(30)
#    subsample=data.imgs.StaticDownsample(30)
    params={'seq_len':30,'dims':(128,64,3),"n_cats":9,
            "read":read,"agum":agum,"subsample":subsample}
    n_frames,n_batch=None,8
    batch_gen=gen.make_batch_gen(in_path,
        n_frames,n_batch,read=params["read"],subsample=params["subsample"])
    generator=gen.AllGenerator(batch_gen,n_cats=params["n_cats"])

    generator=gen.add_agum(generator,params["agum"])

    files.make_dir(out_path)
    nn_path="%s/nn" % out_path
    feat_path="%s/feats" % out_path
    train(generator,nn_path,params,n_epochs)
    extract(in_path,nn_path,feat_path,params)

in_path='../CZU/final'
out_path="lstm"

#single_exp(in_path,out_path,n_epochs=10)
ens(in_path,"lstm",n_cats=22,n_epochs=5)