import sys
sys.path.append("..")
import numpy as np
import data.actions

def make_actions(in_path,out_path):
    data.actions.get_actions(in_path,mean,out_path,dims=None)	

def mean(frames):
    return  np.mean(frames,axis=0)

def diff(frames):
    diff=[ np.abs(frames[i-1]-frames[i]) 
                for i in range(1,len(frames))]	
    return  np.mean(diff,axis=0)

in_path="../../3DHOI2/frames"
out_path="actions"
make_actions(in_path,out_path)