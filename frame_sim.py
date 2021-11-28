import numpy as np
import tensorflow
from tensorflow.keras.layers import Input,Dense,Flatten
from tensorflow.keras.models import Model
from keras.models import load_model
from tensorflow.keras.utils import Sequence
import os.path
import data.actions,data.imgs,data.seqs,data.feats
import sim_core,deep,learn,files,agum,sim_gen

class FrameSim(object):
    def __init__(self,n_hidden=128):
        self.n_hidden=n_hidden
        self.n_kerns=[128,64,64]
        self.kern_size=[(5,5),(3,3),(3,3)]
        self.pool_size=[(4,4),(3,3),(2,2)]  

    def __call__(self,params):
        input_shape=params["input_shape"]
        model,feature_extractor=sim_core.sim_template(input_shape,self)
        model.compile(loss=sim_core.contrastive_loss, optimizer="adam")
        feature_extractor.summary()
        return model,feature_extractor

    def build_model(self,input_shape):
        print(input_shape)
        inputs = Input(input_shape)
        x=deep.add_conv_layer(inputs,self.n_kerns,self.kern_size,
                self.pool_size,one_dim=False)
        x=Flatten()(x)
        x=Dense(self.n_hidden, activation='relu',
            name='hidden',kernel_regularizer=None)(x)
        model = Model(inputs, x)
        return model

def train(sim_gen,out_path,n_epochs=5):#,n_frames=1,n_batch=32):
    make_nn=FrameSim()
    params={"n_cats":9,"input_shape":(128,64,3)    }
    sim_nn=make_nn(params)
#    if(not os.path.exists(out_path)):
#        model=load_model(out_path)
#    else:
    model,extractor=make_nn(params)
    model.fit(sim_gen,epochs=n_epochs)
    if(out_path):
        extractor.save(out_path)

def extract(in_path,nn_path,out_path):
    read=data.imgs.ReadFrames(color="color")
    model=learn.base_read_model(None,nn_path)
    extractor=learn.get_extractor(model,"hidden")
    def helper(frames):
        print(len(frames))
        frames=np.array(frames)
#        frames=np.expand_dims(frames,-1)
        feat_i=extractor.predict(frames)
        return feat_i        
    feat_seq=data.seqs.transform_seqs(in_path,read,helper)
    feat_seq.save(out_path)

def extract_actions(in_path,nn_path,out_path):
    read=data.imgs.ReadFrames(color="color")
    model=learn.base_read_model(None,nn_path)
    extractor=learn.get_extractor(model,"hidden")
    def helper(action_i):
        action_i=np.expand_dims(action_i,0)
        feat_i=extractor.predict(action_i)
        return feat_i 
    agum_fun=[[],agum.flip]
    agum_extr=agum.AgumExtractor(read,helper,agum_fun)
    feat_seq=data.feats.get_feats(in_path,agum_extr)
    feat_seq.save(out_path)

def median(frames):
    return np.median(frames,axis=0)

def sim_exp(in_path,out_path,n_epochs=20,n_batch=32,action=False):
    files.make_dir(out_path)
    nn_path="%s/nn" % out_path
    gen_params={"in_path":in_path,"n_batch":n_batch}
    if(action):
        make_gen=sim_gen.make_action_gen
        extract_fun=extract_actions
    else:
        gen_params["n_frames"]=1
        make_gen=sim_gen.make_sim_gen
        extract_fun=extract
    sim_generator=make_gen(**gen_params)
    agum_fun=[[],agum.flip_sim]
    sim_generator=agum.add_agum(sim_generator,agum_fun)
    train(sim_generator,nn_path,n_epochs=n_epochs)
    seq_path="%s/seqs" % out_path
    extract_fun(in_path,nn_path,seq_path)

in_path="../cc2/segm2/frames"
sim_exp(in_path,"../cc2/segm2",n_epochs=50,action=False)