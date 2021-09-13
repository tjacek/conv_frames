#import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras
from keras.layers import Input,Dense,Flatten,Lambda
from keras.models import Model,Sequential
#import keras
import data.seqs,deep,learn

class TS(object):
    def __init__(self, n_hidden=100,activ='relu',l1=0.01):
        self.activ=activ
        self.n_hidden=n_hidden
        self.l1=l1
        self.n_kerns=[32,32]
        self.kern_size=[8,8]
        self.pool_size=[2,2]
        self.dropout=None

    def __call__(self,params):
        input_img=Input(shape=(params['dims']))
        x=deep.add_conv_layer(input_img,self.n_kerns,self.kern_size,
                            self.pool_size,activ=self.activ,one_dim=True)
        x=deep.full_layer(x,size=self.n_hidden,l1=self.l1,
            dropout=self.dropout,activ=self.activ)
        x=Dense(units=params['n_cats'],activation='softmax')(x)
        model = Model(input_img, x)
        model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam')
        model.summary()
        return model

class TS_SIM(object):
    def __init__(self,n_hidden=100,activ='relu'):
        self.activ=activ
        self.n_hidden=n_hidden
        self.n_kerns=[32,32]
        self.kern_size=[8,8]
        self.pool_size=[2,2]

    def __call__(self,params):
        input_shape=params["input_shape"]
        left_input = Input(input_shape)
        right_input = Input(input_shape)

       
        model = Sequential()

        encoded_l =self.build_model(model,params,input_shape) #model(left_input)
        encoded_r =self.build_model(model,params,input_shape) #model(right_input)

        prediction,loss=contr_loss(encoded_l,encoded_r)
        siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    
        optimizer = keras.optimizers.Adam(lr = 0.00006)
        siamese_net.compile(loss=loss,optimizer=optimizer)
        extractor=Model(inputs=model.get_input_at(0),outputs=model.get_layer("hidden").output)
        extractor.summary()
        return siamese_net,extractor

    def build_model(self,model,params,input):
        deep.add_conv_layer(model,self.n_kerns,self.kern_size,self.pool_size,
                        input,self.activ)
        model.add(Flatten())
        model.add(Dense(self.n_hidden, activation=self.activ,
            name='hidden',kernel_regularizer=None))
        return model

def contr_loss(encoded_l,encoded_r):
    raise Exception(type(encoded_l))
    L2_layer = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)
    return L2_layer([encoded_l, encoded_r]),contrastive_loss

def contrastive_loss(y_true, y_pred):
    margin = 50
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def train_cnn(in_path,out_path,n_epochs=5):
    make_nn=TS_NN()
    read=data.seqs.read_seqs
    train=learn.Train(to_dataset,make_nn,read,batch_size=16)
    train(in_path,out_path,n_epochs)

def train_sim(in_path,out_path,n_epochs=5):
    make_nn=TS_SIM()
    read=data.seqs.read_seqs
    train=learn.SimTrain(read,make_nn)#,read,batch_size=16)
    train(in_path,out_path,n_epochs)

def to_dataset(train):
    X,y=train.to_dataset()
    params={"dims":train.dims(),"n_cats":max(y)+1}	
    return X,y,params

train_sim("seqs","test")