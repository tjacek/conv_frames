import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras
from tensorflow.keras.layers import Input,Dense,Flatten
from tensorflow.keras.models import Model
import learn,data.seqs,deep,sim_core

class TS_SIM(object):
    def __init__(self,n_hidden=100,activ='relu'):
        self.activ=activ
        self.n_hidden=n_hidden
        self.n_kerns=[64,64]
        self.kern_size=[8,8]
        self.pool_size=[2,2]

    def __call__(self,params):
        input_shape=params["input_shape"][1:]
        model=sim_core.sim_template(input_shape,self)
        model.compile(loss=sim_core.contrastive_loss, optimizer="adam")
        feature_extractor.summary()
        return model,feature_extractor

    def build_model(self,input_shape):
        inputs = Input(input_shape)
        x=deep.add_conv_layer(inputs,self.n_kerns,self.kern_size,
                self.pool_size,activ=self.activ,one_dim=True)
        x=Flatten()(x)
        x=Dense(self.n_hidden, activation=self.activ,
            name='hidden',kernel_regularizer=None)(x)
        model = Model(inputs, x)
        return model

def train_sim(in_path,out_path,n_epochs=5):
    make_nn=TS_SIM()
    read=data.seqs.read_seqs
    train=learn.SimTrain(read,make_nn)#,read,batch_size=16)
    train(in_path,out_path,n_epochs)

def extractor(frame_seq,nn_path,out_path):
    read=data.seqs.read_seqs
    extractor=learn.Extract(read)
    extractor(frame_seq,nn_path,out_path)

def sim_exp(in_path,n_epochs=100):
    seq_path="%s/seqs" % in_path
    model_path="%s/model" % in_path
    feat_path="%s/feats" % in_path
    train_sim(seq_path,model_path,n_epochs)
    extractor(seq_path,model_path,feat_path)

in_path="../best2/3_layers"
sim_exp(in_path,n_epochs=100)
