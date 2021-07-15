import numpy as np
import tensorflow.keras
from tensorflow.keras import Input, Model
import data.feats

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
    def __init__(self,make_nn,read=None,read_model=None,name="hidden"):
        self.read=read
        self.make_nn=make_nn
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