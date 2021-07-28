import sys
sys.path.append("..")
import numpy as np
import files

class Paths(dict):
    def __init__(self, arg=[]):
        super(Paths, self).__init__(arg)

    def seqs_len(self):
        return [ len(path_i) for path_i in self.values()]	

    def max_len(self):
    	return max(self.seqs_len())

    def min_len(self):
        return min(self.seqs_len())	

    def supsample(self,size=100):
        for name_i,frames_i in self.items():
            size_i=len(frames_i)
            replace_i= (size_i<size)  
            indexes=np.random.choice(range(size_i),size=size,replace=replace_i)
            indexes.sort()
            self[name_i]=[frames_i[j] for j in indexes ]

def read_paths(in_path):
	paths=Paths()
	for dir_i in files.top_files(in_path):
	    paths[dir_i]=files.top_files(dir_i)
	return paths	

in_path="../../../Downloads/AA/depth/depth_only"
paths=read_paths(in_path)
#print(paths.max_len())
#print(paths.min_len())
paths.supsample()
