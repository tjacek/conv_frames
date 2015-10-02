import os
import os.path as io 

def get_files(path):
    def is_file(f):
        return io.isfile(io.join(path,f))
    all_in_dir=os.listdir(path)
    files= filter(is_file,all_in_dir)
    files.sort()
    return files

def make_dir(path):
    if(not os.path.isdir(path)):
	os.system("mkdir "+path)

def append_path(path,files):
    return map(lambda f: path+f,files)

def replace_sufix(sufix,files):
    return map(lambda s:s.replace(sufix,""),files)
