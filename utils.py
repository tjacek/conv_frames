import os
import os.path as io 
import timeit,pickle,re
import scipy.misc as image

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

def read_file(path):
    file_object = open(path,'r')
    lines=file_object.readlines()  
    file_object.close()
    return lines

def is_file(f,path):
        return io.isfile(io.join(path,f))

def make_dir(path):
    if(not os.path.isdir(path)):
	os.system("mkdir "+path)

def array_to_txt(array):
    return reduce(lambda x,y:x+str(y),array,"")

def save_object(path,nn):
    file_object = open(path,'wb')
    pickle.dump(nn,file_object)
    file_object.close()

def save_string(path,string):
    file_str = open(path,'w')
    file_str.write(string)
    file_str.close()

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

def to_txt_file(path,array):
    txt=array_to_txt(array)
    save_string(path,txt)

def get_zeros(n):
    return [0 for i in range(n)]

def to_csv_file(path,vectors):
    csv=""
    for instance in vectors:
        str_v=vector_to_str(instance)
        str_v+="\n"
        csv+=str_v
    save_string(path,csv)

def to_labeled_file(path,vectors,labels):
    lb=""
    for instance,cat in zip(vectors,labels):
        str_v=vector_to_str(instance)
        str_v+="#"+str(cat)+"\n"
        lb+=str_v
    save_string(path,lb)

def vector_to_str(vector):
    str_vec=""
    size=len(vector)-1
    for i,vec in enumerate(vector):
        vec=round(vec, 2)
        if(i!=size):
            str_vec+=str(vec)+","
        else:
            str_vec+=str(vec)
    return str_vec#+"\n"
