import random
import data.imgs,files

class LazySampler(object):
    def __init__(self,all_paths,read=None,size=30):#,n_batch=8):
        if(read is None):
            read=data.imgs.ReadFrames()	
        self.all_paths=all_paths
        self.read=read
        self.subsample=data.imgs.MinLength(size)
#        self.n_batch=n_batch

    def __len__(self):
        return len(self.all_paths)	

#    def n_iters(self):
#        return int(len(self)/self.n_batch)	

    def get_frames(self,k=100):
        X,y=[],[]
        for path_i in self.get_paths(k):
            name_i=files.get_name(path_i)
            frames=self.read(path_i)
            frames=self.subsample(frames)
#            print(name_i)
#            print(len(frames))
            X.append(frames)
            y.append(name_i.get_cat())
        return X,y

    def get_paths(self,k=100):
        random.shuffle(self.all_paths)
        return self.all_paths[:k]

def make_lazy_sampler(in_path):
    all_paths=files.top_files(in_path)
    all_paths=[ path_i for path_i in all_paths
                    if(is_train(path_i))]
    return LazySampler(all_paths)

def is_train(path_i):
    return (files.get_name(path_i).get_person()==1)

if __name__ == "__main__":
    in_path="../final"
    sampler=make_lazy_sampler(in_path)
    sampler.get_frames(1000)