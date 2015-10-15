import imp
utils =imp.load_source("utils","/home/user/cf/conv_frames/utils.py")
import seq
import simple_bow as bow

def extract_unique(in_path):
    bow.get_labeled_vectors(in_path,compute_unique,".db")

def compute_unique(dataset):
    return map(create_transaction,dataset.instances)

def create_transaction(instance):
    tran={}
    for char in instance.seq:
        tran[char]=True
    s_tran=""
    for key in tran:
        s_tran+=key
    return s_tran

if __name__ == "__main__":
    path="/home/user/cf/seqs/dataset.seq"
    extract_unique(path)
