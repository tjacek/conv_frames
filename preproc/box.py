import sys
sys.path.append("..")
import numpy as np
import cv2,json
import data.actions,files

def get_box(action_path,out_path):
    dataset={}
    for i,path_i in enumerate(files.top_files(action_path)):
        name_i=files.Name(path_i.split("/")[-1]).clean()
        print(name_i)
        img_i=cv2.imread(path_i,cv2.IMREAD_GRAYSCALE)
        x,y,w,h=cv2.boundingRect(img_i)
        dataset[name_i]=(y,x,h,w)
    with open(out_path, 'w') as f:
        json.dump(dataset, f)

action_path="../../actions2"
get_box(action_path,"train")