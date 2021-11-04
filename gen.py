import random
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
import data.imgs,files

class BatchGenerator(object):
    def __init__(self,sampler,n_iters,n_frames=100,n_batch=8):
        self.sampler=sampler
        self.n_iters=n_iters
        self.n_frames=n_frames
        self.n_batch=n_batch
        self.X=None
        self.y=None
        self.i=0
    
    def set(self,X,y):
        self.X=X
        self.y=y

    def get_batch(self, index):
        y_i=self.y[self.i*self.n_batch:(self.i+1)*self.n_batch]
        X_i=self.X[self.i*self.n_batch:(self.i+1)*self.n_batch]
        X_i=np.expand_dims(X_i,axis=-1)
        self.i=(self.i+1) % self.n_iters
        return X_i,y_i

class BinaryGenerator(Sequence):
    def __init__(self,cat,batch_gen):
        self.cat=cat
        self.batch_gen=batch_gen
        self.on_epoch_end()

    def __len__(self):
        return self.batch_gen.n_iters

    def on_epoch_end(self):
        if(self.batch_gen.i==0):
            sampler=self.gen_batch.sampler
            in_paths=sampler.get_category(self.cat)
            selector=lambda path_j: get_cat(path_j)!=self.cat
            out_paths=sampler.get_paths(self.n_frames,selector)
            paths=in_paths+out_paths
            X,y=sampler.get_frames(paths)
            y= [ int(self.cat==y_k) for y_k in y]
            y=to_categorical(y,2)
            self.batch_gen.set(X,y)

    def __getitem__(self, index):
        return self.get_batch(index)

class AllGenerator(Sequence):   
    def __init__(self,batch_gen):
        self.batch_gen=batch_gen
        self.on_epoch_end()

    def __len__(self):
        return self.batch_gen.n_iters

    def on_epoch_end(self):
        if(self.i==0):
            X,y=self.sampler.get_frames(self.n_frames)
            X=np.array(X)
            y=to_categorical(y,12)
            self.batch_gen.set(X,y)

    def __getitem__(self, index):
        return self.get_batch(index)

class LazySampler(object):
    def __init__(self,all_paths,read=None,size=30):
        if(read is None):
            read=data.imgs.ReadFrames()	
        self.all_paths=all_paths
        self.read=read
        self.subsample=data.imgs.MinLength(size)

    def __len__(self):
        return len(self.all_paths)	

    def get_frames(self,paths=100):
        X,y=[],[]
        if(type(paths)==int):
            paths=self.get_paths(k)
        for path_i in paths:
            name_i=files.get_name(path_i)
            frames=self.read(path_i)
            frames=self.subsample(frames)
            X.append(frames)
            y.append(name_i.get_cat())
        return X,y

    def get_paths(self,k=100,selector=None):
        if(selector):
            paths=[]
            for path_j in self.all_paths:
                if(selector(path_j)):
                    paths.append(path_j)
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

def make_lazy_sampler(in_path):
    all_paths=files.top_files(in_path)
    all_paths=[ path_i for path_i in all_paths
                    if(is_train(path_i))]
    return LazySampler(all_paths)

def get_cat(path_i):
    return files.get_name(path_i).get_cat()

def is_train(path_i):
    return (files.get_name(path_i).get_person()==1)

if __name__ == "__main__":
    in_path="../final"
    sampler=make_lazy_sampler(in_path)
    sampler.get_frames(1000)