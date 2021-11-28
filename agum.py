import numpy as np
from tensorflow.keras.utils import Sequence
import files

class AgumDecorator(Sequence):
    def __init__(self,generator,agum):
        self.generator=generator
        self.agum=agum

    def __len__(self):
        return len(self.agum)*len(self.generator)

    def on_epoch_end(self):
        self.generator.on_epoch_end()

    def __getitem__(self, index):
        agum_index=index % len(self.agum)
        batch_index= int(index / len(self.agum))
        X,y=self.generator[batch_index]
        agum_k=self.agum[agum_index]
        X_agum=apply_agum(X,agum_k)
        return X_agum,y

def apply_agum(X,agum_i):
    if(type(agum_i)!=list):
        return agum_i(X)
    for fun_j in agum_i:
        X=fun_j(X)
    return X

def add_agum(generator,agum):
    return AgumDecorator(generator,agum)    

class AgumExtractor(object):
    def __init__(self,preproc,extract,agum,selector=None):
        if(selector is None):
            selector=files.person_selector
        self.preproc=preproc
        self.extract=extract
        self.agum=agum
        self.selector=selector

    def __call__(self,in_path):
        print(in_path)
        seq_i=self.preproc(in_path)
        name_i=files.get_name(in_path)
        if(self.selector(name_i)):
            all_agum=[]
            for agum_j in self.agum:
                agum_seq=apply_agum(seq_i,agum_j)
                all_agum.append(self.extract(agum_seq))
            return all_agum
        else:
            return self.extract(seq_i)

def flip(frames):
    frames= [np.flip(frame_i,axis=1) 
                for frame_i in frames]
    return frames

def flip_img(img_i):
    raise Exception(img_i[0].shape)
    return np.flip(img_i,axis=1) 

def flip_sim(img_i):
    a,b=img_i
    return [np.flip(a,axis=1),b]

def reverse(frames):
    frames=np.array(frames)
    frames= np.flip(frames,axis=0)
    return frames