import sys
sys.path.append("..")
import numpy as np
import data.actions

def make_actions(in_path,out_path):
    fun= lambda x: np.mean(x,axis=0)
    data.actions.get_actions(in_path,fun,out_path,dims=None)	

in_path="../../../Downloads/AA/depth/depth_sampled"
out_path="../../../Downloads/AA/depth/actions"
make_actions(in_path,out_path)