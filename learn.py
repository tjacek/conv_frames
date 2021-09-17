import numpy as np
import tensorflow.keras
from tensorflow.keras import Input, Model
import data.feats,data.seqs,sim_core

class Train(object):
    def __init__(self,to_dataset,make_nn,read=None,save_model=None,batch_size=16):
        self.read=read
        self.to_dataset=to_dataset
        self.make_nn=make_nn
        self.batch_size=batch_size
        if(save_model is None):
            save_model=base_save_model
        self.save=save_model

    def __call__(self,frame_seq,out_path,n_epochs=5):
        if(self.read and type(frame_seq)==str):
            frame_seq=self.read(frame_seq)	
        train,test=frame_seq.split()
        X,y,params=self.to_dataset(train)
        if("n_cats" in params ):
            y=to_one_hot(y,params["n_cats"])
        model=self.make_nn(params)
        model.fit(X,y,epochs=n_epochs,batch_size=self.batch_size)
        self.save(model,out_path)

class Extract(object):
    def __init__(self,read=None,read_model=None,name="hidden"):
        self.read=read
        self.name=name
        if(read_model is None):
            read_model=base_read_model
        self.read_model=read_model

    def __call__(self,frame_seq,nn_path,out_path):
        if(self.read and type(frame_seq)==str):
            frame_seq=self.read(frame_seq)	
        model=self.read_model(frame_seq,nn_path)#,self.make_nn)
        extractor=get_extractor(model,self.name)
        feats=get_features(frame_seq,extractor)
        feats.save(out_path)

class ExtractSeqs(object):
    def __init__(self,read,name="hidden"):
        self.read=read
        self.name=name

    def __call__(self,in_path,nn_path,out_path):
        data_dict=self.read(in_path)
        model=base_read_model(data_dict,nn_path)
        extractor=get_extractor(model,self.name)
        def helper(img_i):
            img_i=np.array(img_i)
            feat_i=extractor.predict(img_i)
            return feat_i
        seq_dict=data_dict.transform(helper,new=True,single=False)
        seq_dict=data.seqs.Seqs(seq_dict)
        seq_dict.save(out_path)

class SimTrain(object):
    def __init__(self,read,make_nn,to_dataset,n_batch=8):
        self.read=read
        self.make_nn=make_nn
        self.to_dataset=to_dataset
        self.n_batch=n_batch

    def __call__(self,data_dict,out_path,n_epochs=5):
        if(type(data_dict)==str):
            data_dict=self.read(data_dict ) 
        train,test=data_dict.split()
        X,y,params=self.to_dataset(train)
#        X,y=sim_core.pair_dataset(train)
#        params={"n_cats": max(y)+1, "input_shape":(None,*train.dims())}
        model,extractor=self.make_nn(params)
        model.fit(X,y,epochs=n_epochs)
        if(out_path):
            extractor.save(out_path)

def get_features(frame_seq,extractor):
    feats=data.feats.Feats()
    for i,name_i in enumerate(frame_seq.names()):
        x_i=np.array(frame_seq[name_i])
        x_i=np.expand_dims(x_i,axis=0)
        feats[name_i]= extractor.predict(x_i)
    return feats
   
def base_read_model(frame_seq,nn_path):
    return tensorflow.keras.models.load_model(nn_path)

def base_save_model(model,out_path):
    if(out_path):
        model.save(out_path)

def get_extractor(model,name):
    extractor=Model(inputs=model.input,
                outputs=model.get_layer(name).output)
    extractor.summary()    
    return extractor

def to_one_hot(y,n_cats=20):
    one_hot=np.zeros((len(y),n_cats))
    for i,y_i in enumerate(y):
        one_hot[i,y_i]=1
    return one_hot