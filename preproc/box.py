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
        clean(img_i)
        x,y,w,h=cv2.boundingRect(img_i)
        dataset[name_i]=(y,x,h,w)
    with open(out_path, 'w') as f:
        json.dump(dataset, f)

def clean(img_i,delta=5):
    img_i[:delta,:]=0
    img_i[-delta:,:]=0
    img_i[:,:delta]=0
    img_i[:,-delta:]=0

def basic_actions(in_path,out_path):
    def helper(name_i,frames):
        return np.mean(frames,axis=0)
    data.actions.get_actions_lazy(in_path,out_path,helper)

if __name__ == "__main__":
    action_path="../../cc/box"
#    basic_actions("../../tag",action_path)
    get_box(action_path,"../../cc/train")
    import rect 
    rect.do_exp("../../rgb","../../cc/actions","../../cc/train")