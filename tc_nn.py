import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D
from tensorflow.keras.models import Sequential
from tcn import TCN, tcn_full_summary
import tensorflow.keras.losses
import data.imgs

def make_tcn(params):
    inputs = Input(shape=(params['seq_len'],*params['dims'],1))
    x = Lambda(lambda y: K.reshape(y, (-1, *params['dims'], 1)))(inputs)
    x = Conv2D(16, 5)(x)
    x = MaxPool2D()(x)
    x = Conv2D(16, 5)(x)
    x = MaxPool2D()(x)
    num_features_cnn = np.prod(K.int_shape(x)[1:])
    x = Lambda(lambda y: K.reshape(y, (-1, params['seq_len'], num_features_cnn)))(x)
    x = TCN(2*params['n_cats'])(x)
    x = Dense(params['n_cats'], activation='sigmoid')(x)
    model = Model(inputs=[inputs], outputs=[x])
    model.summary()
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model
#    tcn_layer = TCN(input_shape=(params["ts_len"], params["n_feats"]),name="tcn_layer")
#    m = Sequential([tcn_layer,Dense(units=params['n_cats'],activation='softmax')])
#    m.compile(optimizer='adam', loss=tensorflow.keras.losses.categorical_crossentropy)
#    tcn_full_summary(m, expand_residual_blocks=False)
#    return m

def simple_exp(in_path,seq_size=20):
    frame_seq=data.imgs.read_frame_seqs(in_path,n_split=1)
    frame_seq.subsample(seq_size)
    frame_seq.scale((64,64))
    train,test=frame_seq.split()
    X,y,params=to_dataset(train)
    if("n_cats" in params ):
        y=to_one_hot(y,params["n_cats"])
    model=make_tcn(params)
    model.fit(X,y,epochs=5)
    print(len(frame_seq))

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
simple_exp(in_path)