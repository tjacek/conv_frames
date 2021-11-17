import random
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
import data.imgs,files

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
        X_agum=self.apply_agum(X,agum_k)
        return X_agum,y

    def apply_agum(self,X,agum_i):
        if(type(agum_i)!=list):
            return agum_i(X)
        for fun_j in agum_i:
            X=fun_j(X)
        return X

def add_agum(generator):
    agum=[[],flip,reverse,[flip,reverse]]
    return AgumDecorator(generator,agum)    

#class AgumDecorator(Sequence):
#    def __init__(self,generator,agum=None):
#        if(agum is None):
#            agum=flip
#        self.generator=generator
#        self.agum=agum

#    def __len__(self):
#        return 2*len(self.generator)

#    def on_epoch_end(self):
#        self.generator.on_epoch_end()

#    def __getitem__(self, index):
#        if(index> (len(self.generator )+1)):
#            index-=len(self.generator)
#            X,y=self.generator[index]
#            X=np.array([self.agum(seq_i) for seq_i in X])
#            return X,y
#        else:
#            return self.generator[index]

def flip(frames):
    frames= [np.flip(frame_i,axis=1) 
                for frame_i in frames]
    return frames

def reverse(frames):
    frames=np.array(frames)
    frames= np.flip(frames,axis=0)
#    frames.reverse()
    return frames

class BatchGenerator(object):
    def __init__(self,sampler,n_frames=100,n_batch=8):
        self.sampler=sampler
        self.n_frames=n_frames
        self.n_batch=n_batch
        self.X=None
        self.y=None
        self.i=0

    def n_iters(self):
        return int(self.X.shape[0]/self.n_batch)

    def set(self,X,y):
        print(X.shape)
        self.X=X
        self.y=y
        self.i=0

    def __getitem__(self, raw_index):
        index= raw_index %self.n_iters()
        y_i=self.y[index *self.n_batch:(index+1)*self.n_batch]
        X_i=self.X[index*self.n_batch:(index+1)*self.n_batch]
        if(len(X_i.shape)<5):
            X_i=np.expand_dims(X_i,axis=-1)
        print("\n %d %d \n" % (index,raw_index))
        return X_i,y_i

class BinaryGenerator(Sequence):
    def __init__(self,cat,batch_gen):
        self.cat=cat
        self.batch_gen=batch_gen
        self.on_epoch_end()

    def __len__(self):
        return self.batch_gen.n_iters()

    def on_epoch_end(self):
#        if(self.batch_gen.i==0):
        self.batch_gen.i=0
        sampler=self.batch_gen.sampler
        if( self.batch_gen.n_frames is None):
            paths=self.batch_gen.sampler.all_paths
        else:
            in_paths=sampler.get_category(self.cat)
            selector=lambda path_j: get_cat(path_j)!=self.cat
            out_paths=sampler.get_paths(self.batch_gen.n_frames,selector)
            paths=in_paths+out_paths
        random.shuffle(paths)
        X,y=sampler.get_frames(paths)
        X=np.array(X)
        y= [ int(self.cat==y_k) for y_k in y]
        y=to_categorical(y,2)
        self.batch_gen.set(X,y)

    def __getitem__(self, index):
        return self.batch_gen[index]

class AllGenerator(Sequence):   
    def __init__(self,batch_gen,n_cats=12):
        self.batch_gen=batch_gen
        self.n_cats=n_cats
        self.on_epoch_end()

    def __len__(self):
        return self.batch_gen.n_iters()

    def on_epoch_end(self):
#        if(self.batch_gen.i==0):
        self.batch_gen.i=0
        sampler= self.batch_gen.sampler
        X,y=sampler.get_frames(self.batch_gen.n_frames)
        X=np.array(X)
        y=to_categorical(y, self.n_cats)
        self.batch_gen.set(X,y)

    def __getitem__(self, index):
#        raise Exception(len(self))
        return self.batch_gen[index]

class LazySampler(object):
    def __init__(self,all_paths,read=None,size=30):
        if(read is None):
            read=data.imgs.ReadFrames()	
        self.all_paths=all_paths
        self.read=read
        self.subsample=data.imgs.StaticDownsample(size)#MinLength(size)

    def __len__(self):
        return len(self.all_paths)	

    def get_frames(self,paths=100):
        X,y=[],[]
        if(paths is None):
            paths=self.all_paths
        if(type(paths)==int):
            paths=self.get_paths(paths)
        for path_i in paths:
            name_i=files.get_name(path_i)
            frames=self.read(path_i)
            frames=self.subsample(frames)
            X.append(frames)
            y.append(name_i.get_cat())
        return X,y

    def get_paths(self,k=100,selector=None):
        if(selector):
            paths=self.select_paths(selector)
        else:
            paths=self.all_paths
        random.shuffle(paths)
        return self.all_paths[:k]

    def get_category(self,i):
        cat_i=[]
        for path_j in self.all_paths:
            cat_j=get_cat(path_j)
            if(cat_j==i):
                cat_i.append(path_j)
        return cat_i 

    def select_paths(self,selector):
        paths=[]
        for path_j in self.all_paths:
            if(selector(path_j)):
                paths.append(path_j)
        return paths

def make_batch_gen(in_path,n_frames,n_batch,read="color"):
    sampler=make_lazy_sampler(in_path,read)
    return BatchGenerator(sampler,n_frames,n_batch)

def make_lazy_sampler(in_path,read):
    all_paths=files.top_files(in_path)
    all_paths=[ path_i for path_i in all_paths
                    if(is_train(path_i))]
    if(type(read)==str):
        read=data.imgs.ReadFrames(color=read)
    return LazySampler(all_paths,read=read)

def get_cat(path_i):
    return files.get_name(path_i).get_cat()

def is_train(path_i):
    name_i=files.get_name(path_i)
    return (name_i.get_person()%2)==1

if __name__ == "__main__":
    in_path="../final"
    sampler=make_lazy_sampler(in_path)
    sampler.get_frames(1000)