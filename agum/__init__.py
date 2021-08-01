import sys
sys.path.append("..")
import numpy as np
import data.imgs,files

def agum_seqs(in_path,out_path):
    frame_seqs=data.imgs.read_frame_seqs(in_path,n_split=1)
    frame_seqs.transform(lambda seq_i:seq_i[:50],new=False,single=False)
    agum_func=lambda img_i:np.flip(img_i,axis=0)
    agum_seqs=agum(frame_seqs,agum_func)
    agum_seqs.save(out_path)

def agum(frame_seqs,funcs):
    if(type(funcs)!=list):
        funcs=[funcs]
    agum_dict=data.imgs.FrameSeqs(  frame_seqs.items())
    train=frame_seqs.split()[0]
    for name_i,seq_i in train.items():
        for j,fun_j in  enumerate(funcs):
            name_j=files.Name("%s_%d" % (name_i,j))
            agum_dict[name_j]=fun_j(seq_i)
    return agum_dict

in_path="../../3DHOI2/frames"
out_path="../../3DHOI2/agum"
agum_seqs(in_path ,out_path)