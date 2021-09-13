from keras.layers import Conv2D,Conv1D, MaxPooling1D,MaxPooling2D
from keras.layers import Dropout,Flatten,Dense
from keras import regularizers

def add_conv_layer(model,n_kerns,kern_size,pool_size,
                    input,activ='relu',one_dim=False):

    Conv=Conv1D if(one_dim) else Conv2D
    MaxPooling=MaxPooling1D if(one_dim) else MaxPooling2D
    for i,n_kern_i in enumerate(n_kerns):
        print(i)
        model.add(Conv(filters=n_kern_i, kernel_size=kern_size[i],activation=activ,name='conv%d'%i))
        model.add(MaxPooling(pool_size=pool_size[i],name='pool%d' % i))
    return model

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