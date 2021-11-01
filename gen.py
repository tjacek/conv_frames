import random
import data.imgs,files

class LazySampler(object):
    def __init__(self,all_paths):
    	self.all_paths=all_paths
    
    def __len__(self):
        return len(self.all_paths)	

    def get_paths(self,k=100):
    	random.shuffle(self.all_paths)
    	return self.all_paths[:k]

def make_lazy_sampler(in_path):
    all_paths=files.top_files(in_path)
    all_paths=[ path_i for path_i in all_paths
                    if(is_train(path_i))]
    return LazySampler(all_paths)

def is_train(path_i):
    return files.get_name(path_i).get_person()==1	

in_path="../final"
sampler=make_lazy_sampler(in_path)
print(sampler.get_paths())