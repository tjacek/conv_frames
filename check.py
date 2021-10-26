import numpy as np
from collections import defaultdict
import data.imgs,files

def seq_len(in_path):
    frame_seqs=data.imgs.read_frame_seqs(in_path,n_split=1)
    cats=by_cat(frame_seqs)
    cats_len={ cat_i:[ len(frame_seqs[name_j]) for name_j in names_i] 
        for cat_i,names_i in cats.items()}
    final=[ np.mean(cats_len[name_i]) 
                for name_i in cats_len.keys()]
    print(final)

def by_cat(frame_seqs):
    name_id=defaultdict(lambda :[])
    for name_i in frame_seqs.keys():
        name_id[name_i.get_cat()].append(name_i)
    return name_id	

def split_size(in_path):
    frame_seqs=data.imgs.read_frame_seqs(in_path,n_split=1)
    train,test=frame_seqs.split()
    print("train:%d" % len(train))
    print("test:%d" % len(test))

def mean_value(in_path):
    frame_seqs=data.imgs.read_frame_seqs(in_path,n_split=1)
    cats=by_cat(frame_seqs)
    for cat_i,names_i in cats.items():
        print(cat_i)
        values=[ (name_j,np.mean(frame_seqs[name_j])) 
                    for name_j in names_i]
        mean_i=np.mean([value_i[1] for value_i in values])
        print(mean_i)

def empty_dirs(in_path):
    empty=[]
    for path_i in files.top_files(in_path):
        len_i=len(files.top_files(path_i))
        if(len_i==0):
            empty.append(path_i)
    print(empty)

in_path="../box/frames"
empty_dirs(in_path)