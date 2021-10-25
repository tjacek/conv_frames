import sys
sys.path.append("..")
import numpy as np
from skimage.measure import label
import cv2 
import data.actions,data.imgs

class Backgrounds(object):
    def __init__(self,backgrounds):
        self.backgrounds=backgrounds

    def __call__(self,name_i,action_i):
        print(name_i)
        view_j=name_i.split("_")[-1].split(".")[0]
        background_j=self.backgrounds[view_j]
        diff_i=(action_i-background_j)
        diff_i[diff_i<0]=0
        return diff_i

def substract(in_path,action_path,out_path):
    backgrounds=data.actions.read_actions(in_path,"color")
    backgrounds.transform(to_grey)
    backgrounds=Backgrounds(backgrounds) 
    def helper(name_i,action_i):
        action_i=to_grey(action_i)
        action_i=backgrounds(name_i,action_i)
        action_i=remove_mc(action_i,k=20)
        action_i=largest_cc(action_i)
        return action_i
    data.actions.transform_lazy(action_path,out_path,helper)

def remove_mc(diff_i,k=20):
    u, c = np.unique(diff_i, return_counts=True)
    largest=c.argsort()[-k:]
    for l in largest:
        diff_i[diff_i==u[l]]=0
    return diff_i

def largest_cc(segmentation):
    segmentation[segmentation>0]=1
    labels = label(segmentation)
    largestCC = (labels == (np.argmax(np.bincount(labels.flat)[1:])+1))
    segmentation[  largestCC]=100
    segmentation[segmentation!=100]=0
    return segmentation

def to_grey(frames):
    if(type(frames)==list):
        return [cv2.cvtColor(frame_i, cv2.COLOR_BGR2GRAY)
                    for frame_i in frames
                        if(not (frame_i is None))]
    return cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

#def cv_background(in_path,out_path,background_path):
#    backgrounds=data.actions.read_actions(background_path,"color")
#    backgrounds.transform(to_grey)
#    def helper(name_i,frames):
#        print(name_i)
#        view_j=name_i.split("_")[-1].split(".")[0]
#        frames=to_grey(frames)
#        background_j=backgrounds[view_j]
#        frames=[ (frame_i-background_j) for frame_i in frames]
#        for frame_i in frames:
#            frame_i[frame_i<0]=0
#        return frames        
#    data.imgs.transform_lazy(in_path,out_path,helper)

in_path="../../Background/raw"
action_path="../../actions"
out_path="../../actions2"
substract(in_path,action_path,out_path)
#larges_cc(out_path,"../../actions3")
#cv_background("../../rgb","../../box",in_path)