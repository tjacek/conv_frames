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

def get_actions(in_path,out_path,sufix=".nonzero"): 
    files = get_files(in_path)
    in_paths=append_path(in_path,files)
    actions=replace_sufix(sufix,files)
    make_dir(out_path)
    for i,action in enumerate(actions):
        full_in_path=in_paths[i]
        full_out_path=out_path+action
        make_dir(full_out_path)
        decompose_action(full_in_path,full_out_path,i)

def decompose_action(in_file,out_file,index):
    out_file+="/frame_"+str(index)+"_"
    print(out_file)
    cmd="th decompose_action.lua " + in_file+" "+out_file
    os.system(cmd)

if __name__ == "__main__":
    in_path="/home/user/Desktop/nonzero_data/"
    out_path="/home/user/cf/actions/" 
    get_actions(in_path,out_path,".nonzero")
