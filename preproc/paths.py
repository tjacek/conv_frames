import sys
sys.path.append("..")
import shutil
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

    def sample(self,size=100):
        for name_i,frames_i in self.items():
            size_i=len(frames_i)
            replace_i= (size_i<size)  
            indexes=np.random.choice(range(size_i),size=size,replace=replace_i)
            indexes.sort()
            self[name_i]=[frames_i[j] for j in indexes ]

    def save(self,out_path):
        files.make_dir(out_path)
        for name_i,frames_i in self.items():
            new_dir_i="%s/%s" % (out_path,name_i.split("/")[-1])
            new_dir_i=new_dir_i.replace(".._..","_")
            files.make_dir(new_dir_i)
            print(new_dir_i)
            print(len(frames_i))
            for j,path_j in enumerate(frames_i):
                path_j= path_j.replace(".._..","_")
#                print(path_j)
#                id_j="/".join(path_j.split("/")[-2:])
                id_j=path_j.split("/")[-2]
                name_j="%s/%s/%d" % (out_path,id_j,j)
                print(name_j)
                shutil.copy(path_j,name_j)

def read_paths(in_path):
	paths=Paths()
	for dir_i in files.top_files(in_path):
	    paths[dir_i]=files.top_files(dir_i)
	return paths	

if __name__ == "__main__":
    in_path="../../../Downloads/AA/depth/depth_only"
    out_path="../../../Downloads/AA/depth/depth_sampled"
    paths=read_paths(in_path)
    paths.sample()
    print(paths.seqs_len())
    paths.save(out_path)
