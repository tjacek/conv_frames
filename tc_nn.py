import numpy as np
import data.imgs

def simple_exp(in_path,seq_size=20):
    frame_seq=data.imgs.read_frame_seqs(in_path,n_split=1)
    frame_seq.subsample(seq_size)
    frame_seq.scale((64,64))
    train,test=frame_seq.split()
    X,y,params=to_dataset(train)
    if("n_cats" in params ):
        y=to_one_hot(y,params["n_cats"])
    print(len(frame_seq))

def to_dataset(frames):
    X,y=frames.to_dataset()
    params={'ts_len':X.shape[1],'n_feats':X.shape[2],
                'n_cats':frames.n_cats()}
    return X,y,params

def to_one_hot(y,n_cats=20):
    one_hot=np.zeros((len(y),n_cats))
    for i,y_i in enumerate(y):
        one_hot[i,y_i]=1
    return one_hot

in_path="../MSR/frames"
simple_exp(in_path)