import numpy as np
import data.seqs,data.feats

def simple_feats(in_path,out_path):
    seq_dict= data.seqs.read_seqs(in_path)
    def helper(seq_i):
    	mean_i=np.mean( seq_i,axis=0)
    	std_i=np.std( seq_i,axis=0)
    	feat_i=np.concatenate([mean_i,std_i])
    	return feat_i
    feat_dict=seq_dict.to_feats(helper)
    feat_dict.save(out_path)

simple_feats("seqs","feats")