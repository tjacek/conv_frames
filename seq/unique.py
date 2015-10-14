import imp
utils =imp.load_source("utils","/home/user/cf/conv_frames/utils.py")
import seq

def extract_unique(in_path):
    out_path=in_path.replace(".seq",".db")
    dataset=seq.create_dataset(in_path)
    #labels=dataset.get_labels()
    trans=map(create_transaction,dataset.instances)
    utils.to_txt_file(out_path,trans)

def create_transaction(instance):
    tran={}
    for char in instance.seq:
        tran[char]=True
    s_tran=""
    for key in tran:
        s_tran+=key+","
    return s_tran+"\n"

if __name__ == "__main__":
    path="/home/user/cf/seqs/dataset.seq"
    extract_unique(path)
