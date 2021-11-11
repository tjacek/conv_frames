import sys
sys.path.append("..")
import numpy as np
import cv2
import files,data.imgs,data.actions

def to_rgb(in_path,out_path):
    files.make_dir(out_path)	
    for path_i in files.top_files(in_path):
        postfix=path_i.split(".")[-1]
        if(postfix=="avi"):
            name_i= files.get_name(path_i)
            gest,person,action,cat=name_i.split("_")
            name_i="_".join([cat,person,action,gest])
            print(name_i)	
            vidcap = cv2.VideoCapture(path_i)
            success,frames=True,[]
            while(success):
                success,image = vidcap.read()
                if(success):
                    frames.append(image)                
            out_i="%s/%s" % (out_path, name_i)
            data.imgs.save_frames(out_i,frames)

def subs_background(in_path,out_path):
    fgbg = cv2.createBackgroundSubtractorMOG2()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    def helper(name_i,frames):
        print(name_i) 
        for frame_i in frames:
            fgbg.apply(frame_i)
        masks=[fgbg.apply(frame_i) for frame_i in frames]
        masks = [cv2.morphologyEx(mask_i, cv2.MORPH_OPEN, kernel)
                   for mask_i in masks]
        action_i=np.sum(masks,axis=0)
        return action_i

    data.actions.get_actions_lazy(in_path,out_path,helper)

def simple(in_path,out_path):
    def helper(name_i,frames):
        print(name_i)
        mask_i=np.mean(frames,axis=0)
        frames=[frame_i-mask_i for frame_i in frames]
        return frames
    data.imgs.transform_lazy(in_path,out_path,helper,recreate=True)

def remove_noise(in_path,out_path):
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    def helper(name_i,action_i):
        mask = cv2.morphologyEx(action_i, cv2.MORPH_CLOSE, se1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
        action_i= mask*action_i
        action_i[action_i!=0]=100
        return action_i
    data.actions.transform_lazy(in_path,out_path,helper)

in_path="../../raw"
rgb_path="../../rgb"
box_path="../../box"
action_path="../../actions"
#to_rgb(in_path,out_path)
#subs_background(box_path,action_path)
remove_noise(action_path,"../../actions2")