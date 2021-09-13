import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras
from tensorflow.keras.layers import Input,Dense,Flatten,Lambda
from tensorflow.keras.models import Model
import learn,data.seqs,deep

class TS_SIM(object):
    def __init__(self,n_hidden=100,activ='relu'):
        self.activ=activ
        self.n_hidden=n_hidden
        self.n_kerns=[32,32]
        self.kern_size=[8,8]
        self.pool_size=[2,2]

    def __call__(self,params):
        input_shape=params["input_shape"][1:]
        img_a = Input(shape=input_shape)
        img_b = Input(shape=input_shape)
        feature_extractor = self.build_model(input_shape)
        feats_a = feature_extractor(img_a)
        feats_b = feature_extractor(img_b)
        distance = Lambda(euclidean_distance)([feats_a, feats_b])
        model = Model(inputs=[img_a, img_b], outputs=distance)
        model.compile(loss=contrastive_loss, optimizer="adam")
        feature_extractor.summary()
        return model

    def build_model(self,input_shape):
        inputs = Input(input_shape)
        x=deep.add_conv_layer(inputs,self.n_kerns,self.kern_size,
                self.pool_size,activ=self.activ,one_dim=True)
        x=Flatten()(x)
        x=Dense(self.n_hidden, activation=self.activ,
            name='hidden',kernel_regularizer=None)(x)
        model = Model(inputs, x)
        return model

def euclidean_distance(vectors):
    (feats_a, feats_b) = vectors
    sum_squared = K.sum(K.square(feats_a-feats_b), axis=1,keepdims=True)
    return K.sqrt(K.maximum(sum_squared, K.epsilon()))

def contrastive_loss(y, preds, margin=1):
    y = tf.cast(y, preds.dtype)
    squaredPreds = K.square(preds)
    squaredMargin = K.square(K.maximum(margin - preds, 0))
    loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
    return loss

def train_sim(in_path,out_path,n_epochs=5):
    make_nn=TS_SIM()
    read=data.seqs.read_seqs
    train=learn.SimTrain(read,make_nn)#,read,batch_size=16)
    train(in_path,out_path,n_epochs)

train_sim("seqs","test")