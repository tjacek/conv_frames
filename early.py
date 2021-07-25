import numpy as np
import data.imgs

class EarlyReader(object):
    def __init__(self,ratio=0.1,dim=(64,64),n_split=1,):
        self.ratio=ratio
        self.dim=dim
        self.n_split=n_split

    def __call__(self,in_path):
        frame_seqs=data.imgs.read_frame_seqs(in_path,n_split=self.n_split)
        mean_size= np.median(frame_seqs.seqs_len())
        seq_len=self.get_fraction(mean_size) 
        sample=data.imgs.MinLength(seq_len)
        def helper(seq_i):
            start_i= self.get_fraction(len(seq_i))
            start_seq_i=seq_i[:start_i]
            final_seq_i=sample(start_seq_i)
            return final_seq_i
        frame_seqs.transform(helper,new=False,single=False)
        return frame_seqs

    def get_fraction(self,len_i):
        return int(np.ceil(len_i*self.ratio))

in_path="../../2021_II/clean3/base/frames"
read=EarlyReader()
read(in_path)