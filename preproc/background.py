import sys
sys.path.append("..")
import numpy as np
from skimage.measure import label
import cv2 
import data.actions,data.imgs

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

def larges_cc(in_path,out_path):
    def helper(name_i,action_i):
        segmentation=np.sum(action_i,axis=2)
        segmentation[segmentation>0]=1
        labels = label(segmentation)
        largestCC = (labels == (np.argmax(np.bincount(labels.flat)[1:])+1))
        segmentation[  largestCC]=100
        return segmentation
    data.actions.transform_lazy(in_path,out_path,helper)

def cv_background(in_path,out_path):
#    backgrounds=data.actions.read_actions(in_path,"color")
    def helper(name_i,frames):
        print(name_i)
#        view_j=name_i.split("_")[-1].split(".")[0]
        frames=[cv2.cvtColor(frame_i, cv2.COLOR_BGR2GRAY)
                    for frame_i in frames
                        if(not (frame_i is None))]
        return frames        
    data.imgs.transform_lazy(in_path,out_path,helper)

in_path="../../Background/raw"
action_path="../../actions"
out_path="../../actions2"
#substract(in_path,action_path,out_path)
#larges_cc(out_path,"../../actions3")
cv_background("../../rgb","../../box")