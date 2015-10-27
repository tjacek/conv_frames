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
    persons=a_dataset.get_persons()
    utils.to_labeled_file(out_path,united_vec,labels)
    return united_vec,persons,labels

def split(a_path,b_path,out_path):
    united_vec,persons,labels=unify(a_path,b_path,out_path)
    dataset_odd=[[],[]]
    dataset_even=[[],[]]
    for i,p_i in enumerate(persons):
        if((p_i % 2) ==1):
            dataset_odd[0].append(united_vec[i])
            dataset_odd[1].append(labels[i])
        else:
            dataset_even[0].append(united_vec[i])
            dataset_even[1].append(labels[i])
    train_path=utils.change_postfix(out_path,"_out.lb",new="_train.lb")
    test_path=utils.change_postfix(out_path,"_out.lb",new="_test.lb")
    utils.to_labeled_file(train_path,dataset_odd[0],dataset_odd[1])
    utils.to_labeled_file(test_path,dataset_even[0],dataset_even[1])

if __name__ == "__main__":
   path="/home/user/cf/seqs/dataset"
   a_path=path+"A.seq"
   b_path=path+"B.seq"
   out_path=path+"_out.lb"
   split(a_path,b_path,out_path)
