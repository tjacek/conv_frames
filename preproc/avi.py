import sys
sys.path.append("..")
import numpy as np
import cv2
import background
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
        action_i= background.largest_cc(action_i)
        return action_i

    data.actions.get_actions_lazy(in_path,out_path,helper)

def simple(in_path,out_path):
    def helper(name_i,frames):
        print(name_i)
        frames= background.to_grey(frames) 
        frames=[ cv2.medianBlur(frame_i, 25) 
                    for frame_i in frames]
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

def detect_circles(in_path,out_path):
    minDist,param1,param2 = 100,30,50  
    minRadius,maxRadius = 5,20
    def helper(name_i,frame_i):
        gray = cv2.cvtColor(frame_i, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 25)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 
            minDist, param1=param1, param2=param2, 
            minRadius=minRadius, maxRadius=maxRadius)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                cv2.circle(frame_i, (i[0], i[1]), i[2], (0, 255, 0), 2)
        return frame_i
    data.imgs.transform_lazy(in_path,out_path,helper,single=True)

def find_tag(in_path,out_path):
    def helper(name_i,frame_i):
        frame_i[:,:,1][frame_i[:,:,0]!=0]=0
        frame_i[:,:,1][frame_i[:,:,0]==0]=1 
        frame_i[:,:,2][frame_i[:,:,2]==0]=0
        tag=frame_i[:,:,1]*frame_i[:,:,2]
        tag=cv2.medianBlur(tag,5)
        return tag
    data.imgs.transform_lazy(in_path,out_path,helper,
         recreate=True,single=True)

in_path="../../raw"
rgb_path="../../rgb"
#tag_path
box_path="../../box"
action_path="../../actions"
#to_rgb(in_path,out_path)
#simple(rgb_path,box_path)
subs_background(rgb_path,"../../cc/box")
#remove_noise(action_path,"../../actions2")
#find_tag(rgb_path,"../../tag")