import os
import os.path as io 
import scipy.misc as image

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

def is_file(f,path):
        return io.isfile(io.join(path,f))

def make_dir(path):
    if(not os.path.isdir(path)):
	os.system("mkdir "+path)

def append_path(path,files):
    return map(lambda f: path+f,files)

def replace_sufix(sufix,files):
    return map(lambda s:s.replace(sufix,""),files)

def read_images(files):
    return map(lambda f:image.imread(f),files)
