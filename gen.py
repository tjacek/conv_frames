import random
import data.imgs,files

class AllGenerator(object):   
    def __init__(self,sampler,n_iters,n_batch=8):
        self.sampler=sampler
        self.n_iters=n_iters
        self.n_batch=n_batch

    def __iter__(self):
        return self

    def __next__(self):
        i,X,y=0,None,None
        while(True):
            if(i==0):
                X,y=self.sampler.get_frames(n_frames)
                y=to_categorical(y,12)  
            y_i=y[i*n_batch:(i+1)*n_batch]
            X_i=X[i*n_batch:(i+1)*n_batch]
            X_i=np.array(X_i)
            X_i=np.expand_dims(X_i,axis=-1)
            i=(i+1) % self.n_iters
            yield X_i,y_i

class LazySampler(object):
    def __init__(self,all_paths,read=None,size=30):
        if(read is None):
            read=data.imgs.ReadFrames()	
        self.all_paths=all_paths
        self.read=read
        self.subsample=data.imgs.MinLength(size)

    def __len__(self):
        return len(self.all_paths)	

    def get_frames(self,k=100):
        X,y=[],[]
        for path_i in self.get_paths(k):
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
            cat_j=get_cat(path_j):
            if(name_j.get_cat()==i):
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