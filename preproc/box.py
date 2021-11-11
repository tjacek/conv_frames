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

def basic_actions(in_path,out_path):
    def helper(name_i,frames):
        return np.mean(frames,axis=0)
    data.actions.get_actions_lazy(in_path,out_path,helper)

in_path="../../tag"
action_path="../../actions2"

#basic_actions(in_path,action_path)
#get_box(action_path,"train2")
basic_actions("../../box/frames","../../actions4")
