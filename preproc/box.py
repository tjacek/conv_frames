import sys
sys.path.append("..")
import numpy as np
import cv2,json
import data.actions

def get_box(action_path,out_path):
    actions=data.actions.read_actions(action_path,"grey")
    dataset={}
    for name_i,img_i in actions.items():
        x,y,w,h=cv2.boundingRect(img_i)
        dataset[name_i]=(y,x,h,w)
    with open(out_path, 'w') as f:
        json.dump(dataset, f)

action_path="../../actions2"
get_box(action_path,"train")