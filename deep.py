import tensorflow.keras.backend as K
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D,Conv1D, MaxPooling1D,MaxPooling2D
from tensorflow.keras.layers import Dropout,Flatten,Dense
from tensorflow.keras import regularizers

def add_conv_layer(model,n_kerns,kern_size,pool_size,
                    activ='relu',one_dim=False):
    x=model
    Conv=Conv1D if(one_dim) else Conv2D
    MaxPooling=MaxPooling1D if(one_dim) else MaxPooling2D
    for i,n_kern_i in enumerate(n_kerns):
        print(i)
        kern_i,pool_i,name_i=kern_size[i],pool_size[i],'conv%d'%i
        x=Conv(filters=n_kern_i,kernel_size=kern_i,activation=activ,name=name_i)(x)
        x=MaxPooling(pool_size=pool_i,name='pool%d' % i)(x)
    return x

def full_layer(x,size=100,l1=0.01,dropout=0.5,activ='relu'):
    x=Flatten()(x)
    reg=regularizers.l1(l1) if(l1) else None
    name="prebatch" if(dropout=="batch_norm") else "hidden"
    if(type(size)==list):
        for size_i in size[:-1]:
            x=Dense(size_i, activation=activ,kernel_regularizer=None)(x)
        size=size[-1]
    x=Dense(size, activation=activ,name=name,kernel_regularizer=reg)(x)
    if(dropout=="batch_norm"):
        return BatchNormalization(name="hidden")(x)
    if(dropout):
        return Dropout(dropout)(x)