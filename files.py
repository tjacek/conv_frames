import os,re,shutil

class Name(str):
    def __new__(cls, p_string):
        return str.__new__(cls, p_string)

    def clean(self):
        digits=[ str(int(digit_i)) 
                for digit_i in re.findall(r'\d+',self)]
        return Name("_".join(digits))

    def get_cat(self):
        return int(self.split('_')[0])-1

    def get_person(self):
        return int(self.split('_')[1])

    def sub_seq(self,k):
        return Name("_".join(self.split("_")[:k]))

def get_name(path_i):
    return Name(path_i.split("/")[-1]).clean()

def top_files(path):
    paths=[ path+'/'+file_i for file_i in os.listdir(path)]
    paths=sorted(paths,key=natural_keys)
    return paths

def bottom_files(path,full_paths=True):
    all_paths=[]
    for root, directories, filenames in os.walk(path):
        if(not directories):
            for filename_i in filenames:
                path_i= root+'/'+filename_i if(full_paths) else filename_i
                all_paths.append(path_i)
    all_paths.sort(key=natural_keys)        
    return all_paths

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def atoi(text):
    return int(text) if text.isdigit() else text

def split(dict,selector=None):
    if(not selector):
        selector=person_selector
    train,test=[],[]
    for name_i in dict.keys():
        if(selector(name_i)):
            train.append((name_i,dict[name_i]))
        else:
            test.append((name_i,dict[name_i]))
    return train,test

def person_selector(name_i):
    person_i=int(name_i.split('_')[1])
    return person_i%2==1

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

def get_paths(dir_path,sufixes):
    return {sufix_i:"%s/%s"%(dir_path,sufix_i) for sufix_i in sufixes}

def prepare_dirs(basic_path,sub_dirs):
    make_dir(basic_path)
    paths=get_paths(basic_path,sub_dirs)
    return paths

def ens_paths(dir_path,common_path,binary_path):
    binary="%s/%s" % (dir_path,binary_path)
    common=["%s/%s" % (dir_path,path_i) for path_i in common_path]
    return {"common":common ,"binary":binary}

def find_dirs(in_path,pred=None):
    if(pred is None):
        pred=lambda x:True
    all_paths=[]
    for root, directories, filenames in os.walk(in_path):
        if(pred(root)):
            all_paths.append(root)
    return all_paths

def move_dirs(paths,out_path,get_name):
    make_dir(out_path)
    for path_i in paths:
        name_i=get_name(path_i)
        out_i="%s/%s" % (out_path,name_i)
        print(out_i)
        shutil.move(path_i,out_i)