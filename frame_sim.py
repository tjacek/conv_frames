import numpy as np
from tensorflow.keras.layers import Input,Dense,Flatten
from tensorflow.keras.models import Model
from keras.models import load_model
import data.actions,data.imgs,data.seqs
import sim_core,deep,learn,files

class FrameSim(object):
    def __init__(self,n_hidden=128):
        self.n_hidden=n_hidden
        self.n_kerns=[64,32]
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

def train(in_path,out_path,n_epochs=5,dims=(128,64)):
    make_nn=FrameSim()
    def read(in_path):
        paths=read_paths(in_path,["1","2","3"])
        print(paths)
        data_dict=data.actions.from_paths(paths)
        data_dict.add_dim()
        return data_dict
    train_sim=learn.SimTrain(read,make_nn,n_batch=8)
    train_sim(in_path,out_path,n_epochs)

def extract(in_path,nn_path,out_path):
    read=data.imgs.ReadFrames()
    model=learn.base_read_model(None,nn_path)
    extractor=learn.get_extractor(model,"hidden")
    def helper(frames):
        print(len(frames))
        frames=np.array(frames)
        frames=np.expand_dims(frames,-1)
        feat_i=extractor.predict(frames)
        return feat_i
    feat_seq=data.seqs.Seqs()
    for i,path_i in enumerate(files.top_files(in_path)):
        frames=[ read(path_j) 
                for path_j in files.top_files(path_i)]
        name_i=files.get_name(path_i)
        feat_seq[name_i]=helper(frames)
    feat_seq.save(out_path)

def center_frame(frames):
    center= int(len(frames)/2)
    return frames[center]

def median(frames):
    return np.median(frames,axis=0)

def get_frames(in_path,out_path,fun=None):
    if(fun is None):
        fun=center_frame
    data.actions.get_actions_eff(in_path,fun,out_path,dims=None)

def read_paths(in_path,persons):
    persons=set(persons)
    paths=[] 
    for path_i in files.top_files(in_path):
        name_i=files.Name(path_i.split("/")[-1]).clean()
        if( (name_i.get_person() %2)==1):
            person_i= name_i.split("_")[2]
            if(person_i in persons):
                paths.append(path_i)
    return paths

def sim_exp(in_path,out_path):
    files.make_dir(out_path)
    get_frames(in_path,"%s/frames" % out_path)

in_path="../final"
sim_exp(in_path,"../center")
#median(in_path,"../median")
#print(len( read_paths("../median")) )
#train("../median",nn_path)
#extract(in_path,nn_path,out_path)