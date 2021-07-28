import sys
sys.path.append("..")
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

def read_paths(in_path):
	paths=Paths()
	for dir_i in files.top_files(in_path):
	    paths[dir_i]=files.top_files(dir_i)
	return paths	

in_path="../../../2021_VI/raw_3DHOI/3DHOI/frames"
paths=read_paths(in_path)
print(paths.max_len())
print(paths.min_len())