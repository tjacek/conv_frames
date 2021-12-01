import numpy as np
import  cv2
import data.imgs,data.actions

def diff(in_path,out_path):
    def fun(name_i,frames):
        print(name_i)
        frames=to_grey(frames)
        size=len(frames)
        return [ np.abs(frames[i]-frames[i-1]) 
                    for i in range(1,size)]
    data.imgs.transform_lazy(in_path,out_path,fun,single=False)

def get_frames(in_path,out_path,fun=None):
    if(fun is None):
        fun=detect_edges
    data.actions.get_actions_lazy(in_path,out_path,fun,read=None)

def get_seqs(in_path,out_path,fun=None):
    if(fun is None):
        fun=get_edges    
    data.imgs.transform_lazy(in_path,out_path,fun,single=True)

def detect_edges(name,frames):
    final=[]
    for frame_i in frames:
        final.append(get_edges(name_i,frame_i))
    action_img=np.mean(final,axis=0)
    action_img[action_img!=0]=100
    return action_img

def get_edges(name_i,frame_i):
    import cv2    
    frame_i=cv2.cvtColor(frame_i, cv2.COLOR_BGR2GRAY)
    frame_i=cv2.medianBlur(frame_i,5)
    frame_i=cv2.Canny(frame_i , 100, 200)
    return frame_i

def to_grey(frames):
    return [cv2.cvtColor(frame_i, cv2.COLOR_BGR2GRAY)
                    for frame_i in frames
                        if(not (frame_i is None))]

def to_grey_transform(in_path,out_path):
    def helper(name_i,frames):
        print(name_i)
        return to_grey(frames)
    data.imgs.transform_lazy(in_path,out_path,helper,single=False)

in_path="../cc2/final"
out_path="../../2021_XII/final"
to_grey_transform(in_path,out_path)