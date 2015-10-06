import os
import os.path as io 
import timeit,pickle
import scipy.misc as image
import load

class Timer(object):
    def __init__(self):
        self.start_time=timeit.default_timer()

    def stop(self):
        self.end_time = timeit.default_timer()
        self.total_time = (self.end_time - self.start_time)

    def show(self):
        print("Training time %d ",self.total_time)

def get_files(path):
    all_in_dir=os.listdir(path)
    files= filter(lambda f:is_file(f,path),all_in_dir)
    files.sort()
    return files

def get_dirs(path):
    all_in_dir=os.listdir(path)
    dirs= filter(lambda f: not is_file(f,path),all_in_dir)
    dirs.sort()
    return dirs

def get_paths(dir_path):
    files=get_files(dir_path)
    files=["/" + f for f in files]
    return append_path(dir_path,files)

def is_file(f,path):
        return io.isfile(io.join(path,f))

def make_dir(path):
    if(not os.path.isdir(path)):
	os.system("mkdir "+path)

def save_object(path,nn):
    file_object = open(path,'wb')
    pickle.dump(nn,file_object)
    file_object.close()

def read_object(path):
    file_object = open(path,'r')
    obj=pickle.load(file_object)  
    file_object.close()
    return obj

def append_path(path,files):
    return map(lambda f: path+f,files)

def replace_sufix(sufix,files):
    return map(lambda s:s.replace(sufix,""),files)

def read_images(files):
    return map(lambda f:image.imread(f),files)
