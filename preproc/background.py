import sys
sys.path.append("..")
import numpy as np
import data.actions

def substract(in_path,action_path,out_path):
    backgrounds=data.actions.read_actions(in_path,"color")
    def helper(name_i,action_i):
        print(name_i)
        view_j=name_i.split("_")[-1].split(".")[0]
        background_j=backgrounds[view_j]
        diff_i=(action_i-background_j)
        diff_i[diff_i<0]=0
        u, c = np.unique(diff_i, return_counts=True)
        largest=c.argsort()[-20:]
        for l in largest:
            diff_i[diff_i==u[l]]=0
        return diff_i
    data.actions.transform_lazy(action_path,out_path,helper)

in_path="../../Background/raw"
action_path="../../actions"
out_path="../../actions2"
substract(in_path,action_path,out_path)