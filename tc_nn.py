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
import data.imgs,data.feats,files

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
        x = TCN(self.n_hidden,name="hidden",use_batch_norm=True,dropout_rate=0.5)(x)
        x = Dense(params['n_cats'], activation='sigmoid')(x)
        model = Model(inputs=[inputs], outputs=[x])
#       model.summary()
        tcn_full_summary(model, expand_residual_blocks=False)
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
        return model

def single_exp(in_path,out_path,n_epochs=100):
    paths=files.prepare_dirs(out_path,["nn","feats"])
    print(paths)
    single_train(in_path,paths["nn"],n_epochs=n_epochs)
    single_extract(in_path,paths["nn"],paths["feats"])

def single_train(in_path,out_path=None,seq_size=20,n_epochs=5):
    frame_seq=data.imgs.read_frame_seqs(in_path,n_split=1)
    frame_seq.subsample(seq_size)
    frame_seq.scale((64,64))
    train,test=frame_seq.split()
    X,y,params=to_dataset(train)
    if("n_cats" in params ):
        y=to_one_hot(y,params["n_cats"])
    make_tcn=TC_NN()
    model=make_tcn(params)
    model.fit(X,y,epochs=n_epochs,batch_size=16)
    if(out_path):
        model.save_weights(out_path)

def single_extract(in_path,nn_path,out_path):
    frame_seq=data.imgs.read_frame_seqs(in_path,n_split=1)
    frame_seq.subsample(20)
    frame_seq.scale((64,64))
    params={'seq_len':20,'dims':(64,64),'n_cats':frame_seq.n_cats()}
    make_tcn=TC_NN()
    model=make_tcn(params)   
    model.load_weights(nn_path)
    extractor=Model(inputs=model.input,
                outputs=model.get_layer("hidden").output)
    extractor.summary()
    feats=data.feats.Feats()
    for i,name_i in enumerate(frame_seq.names()):
        x_i=np.array(frame_seq[name_i])
        x_i=np.expand_dims(x_i,axis=0)
        feats[name_i]= extractor.predict(x_i)
    feats.save(out_path)

def to_dataset(frames):
    X,y=frames.to_dataset()
    params={'seq_len':X.shape[1],'dims':(X.shape[2],X.shape[3]),
                'n_cats':frames.n_cats()}
    return X,y,params

def to_one_hot(y,n_cats=20):
    one_hot=np.zeros((len(y),n_cats))
    for i,y_i in enumerate(y):
        one_hot[i,y_i]=1
    return one_hot

in_path="../MSR/frames"
single_exp(in_path,"drop")