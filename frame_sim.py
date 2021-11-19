import numpy as np
import tensorflow
from tensorflow.keras.layers import Input,Dense,Flatten
from tensorflow.keras.models import Model
from keras.models import load_model
from tensorflow.keras.utils import Sequence
import os.path
import data.actions,data.imgs,data.seqs,data.feats
import sim_core,deep,learn,files,agum

class SimGen(Sequence):
    def __init__(self,sampler,n_frames=3,batch_size=32):
        self.sampler=sampler
        self.n_frames=n_frames
        self.names=sim_core.all_pairs(list(sampler.keys()))
        self.batch_size=batch_size
        self.n_batch=int(len(self.names)/self.batch_size)

    def __len__(self):
        return self.n_frames*self.n_batch
        
    def __getitem__(self, i):
        i= i % self.n_batch
        pairs_i=self.names[i*self.batch_size:(i+1)*self.batch_size]
        X,y=[],[]
        for name_a,name_b in pairs_i:
            x_a=self.sampler[name_a]
            x_b=self.sampler[name_b]
            X.append((x_a,x_b))
            y.append(sim_core.all_cat(name_a,name_b))
        X=np.array(X)
        X=[X[:,0],X[:,1]]
        return X,np.array(y)

class FrameSampler(object):
    def __init__(self,path_dict,dist=None,read="color"):
        if(type(read)==str):
            read=data.imgs.ReadFrames(color=read)
        if(dist is None):
            dist=nonuniform_dist
        self.path_dict=path_dict
        self.read=read
        self.dist=dist

    def __call__(self,name_j):
        seq_j=self.path_dict[name_j]
        index=self.dist(seq_j)
        frame_path=seq_j[index]
        return self.read(frame_path)

def make_action_gen(in_path,n_batch=8):
    action_dict=data.actions.read_actions(in_path,img_type="color")
    action_dict=action_dict.split()[0]
    return SimGen(action_dict,1,n_batch)

def make_sim_gen(in_path,n_frames,n_batch=32):
    paths=files.get_path_dict(in_path)
    train= dict(files.split(paths)[0])
    sampler=FrameSampler(path_dict)
    return SimGen(sampler,n_frames,n_batch)

def uniform_dist(seq_j):
    return np.random.randint(len(seq_j),size=None)

def center_dist(seq_j):
    return int(len(seq_j)/2)

def nonuniform_dist(seq_i):
    size=len(seq_i)
    inc,dec=np.arange(size),np.flip(np.arange(size))
    dist=np.amin(np.array([inc,dec]),axis=0)
    dist=dist.astype(float)
    dist=dist**2
    if(np.sum(dist)==0):
        dist.fill(1.0)
    dist/=np.sum(dist)
    i=np.random.choice(np.arange(size),1,p=dist)[0]
    return i

class FrameSim(object):
    def __init__(self,n_hidden=128):
        self.n_hidden=n_hidden
        self.n_kerns=[128,64]
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
    if(not os.path.exists(out_path)):
        model=load_model(out_path)
    else:
        model,extractor=make_nn(params)
    model.fit(sim_gen,epochs=n_epochs)
    if(out_path):
        model.save(out_path)

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
    def helper(path_i):
        action_i=read(path_i)
        action_i=np.expand_dims(action_i,0)
        feat_i=extractor.predict(action_i)
        return feat_i 
    feat_seq=data.feats.get_feats(in_path,helper)
    feat_seq.save(out_path)

def median(frames):
    return np.median(frames,axis=0)

def diff(name,frames):
    import cv2
    final=[]
    for frame_i in frames:
        frame_i=cv2.cvtColor(frame_i, cv2.COLOR_BGR2GRAY)
        frame_i=cv2.medianBlur(frame_i,5)
        frame_i=cv2.Canny(frame_i , 100, 200)
        final.append(frame_i)
    action_img=np.mean(final,axis=0)
    action_img[action_img!=0]=100
    return action_img

def get_frames(in_path,out_path,fun=None):
    if(fun is None):
        fun=diff
    data.actions.get_actions_lazy(in_path,out_path,fun,read=None)

def sim_exp(in_path,out_path,n_epochs=20,n_batch=32):
    files.make_dir(out_path)
    nn_path="%s/nn" % out_path
    sim_gen=make_action_gen(in_path,n_batch)
    agum_fun=[[],agum.flip_img]
    sim_gen=agum.add_agum(sim_gen,agum_fun)
    train(sim_gen,nn_path,n_epochs=n_epochs)
    seq_path="%s/seqs" % out_path
    extract_actions(in_path,nn_path,seq_path)

in_path="../cc/florence"
#make_sim_gen(in_path,3)
sim_exp("mean_img","../common2",n_epochs=10)
#get_frames(in_path,"mean_img")