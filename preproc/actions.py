import sys
sys.path.append("..")
import numpy as np
import data.actions

def make_actions(in_path,out_path):
#    fun= lambda x: np.mean(x,axis=0)
    data.actions.get_actions(in_path,diff,out_path,dims=None)	

def diff(frames):
    diff=[ np.abs(frames[i-1]-frames[i]) 
                for i in range(1,len(frames))]	
    return  np.mean(diff,axis=0)

in_path="../../../Downloads/AA/depth/rename"
out_path="../../../Downloads/AA/depth/actions"
make_actions(in_path,out_path)