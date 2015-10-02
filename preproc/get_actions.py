import imp
utils =imp.load_source("utils","/home/user/df/deep_frames/utils.py")


def get_actions(in_path,out_path,sufix=".nonzero"): 
    files =utils.get_files(in_path)
    in_paths=utils.append_path(in_path,files)
    actions=utils.replace_sufix(sufix,files)
    utils.make_dir(out_path)
    for i,action in enumerate(actions):
        full_in_path=in_paths[i]
        full_out_path=out_path+action
        utils.make_dir(full_out_path)
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
