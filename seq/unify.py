import seq,simple_bow as bow
import numpy as np
import imp
utils =imp.load_source("utils","/home/user/cf/conv_frames/utils.py")

def unify(a_path,b_path,out_path):
    a_dataset=seq.create_dataset(a_path)
    b_dataset=seq.create_dataset(b_path)
    a_vec=bow.compute_bow(a_dataset)
    b_vec=bow.compute_bow(b_dataset)
    united_vec=[av +bv for av,bv in zip(a_vec,b_vec)]
    labels=a_dataset.get_labels()
    utils.to_labeled_file(out_path,united_vec,labels)

if __name__ == "__main__":
   path="/home/user/cf/seqs/dataset"
   a_path=path+"A.seq"
   b_path=path+"B.seq"
   out_path=path+"_out.lb"
   unify(a_path,b_path,out_path)
