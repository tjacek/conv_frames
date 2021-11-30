import numpy as np
import random
from tensorflow.keras.utils import Sequence
import sim_core,files,data.imgs

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
    
    def keys(self):
        return self.path_dict.keys()

    def __getitem__(self,name_j):
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
    train= paths.split()[0]#dict(files.split(paths)[0])
    sampler=FrameSampler(train)
    return SimGen(sampler,n_frames,n_batch)

def make_disc_gen(in_path,n_batch=8):
    paths=files.get_path_dict(in_path)
    read=data.imgs.ReadFrames(color="color")
    frame_dict={}
    for name_i,frames_i in paths.items():
        random.shuffle( frames_i)
        paths_i=frames_i[:30]
        for path_j in paths_i:
            name_j=files.get_name(path_j)
            frame_dict[name_j]=read(path_j)
#    frame_dict={str(i):read(path_i) 
#                    for i,path_i in enumerate(s_paths)}
    return SimGen(frame_dict,1,n_batch)
#    raise Exception(len(frame_dict))

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