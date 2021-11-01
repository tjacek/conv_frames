import random
import data.imgs,files

class LazySampler(object):
    def __init__(self,all_paths,read=None):
        if(read is None):
            read=data.imgs.ReadFrames()	
        self.all_paths=all_paths
        self.read=read 

    def __len__(self):
        return len(self.all_paths)	

    def get_frames(self,k=100):
        X,y=[],[]
        for path_i in self.get_paths(k):
            name_i=files.get_name(path_i)
            frames=self.read(path_i)
            print(name_i)
            print(len(frames))

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

in_path="../final"
sampler=make_lazy_sampler(in_path)
sampler.get_frames()