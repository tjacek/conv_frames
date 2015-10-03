import imp
utils =imp.load_source("utils","/home/user/cf/conv_frames/utils.py")
import numpy as np

class LabeledImages(object):
    def __init__(self,images,labels,n_cat):
        self.X=images
        self.y=labels
        self.n_cat=n_cat

def read_images(path):
    cat_dirs=utils.get_dirs(path)
    n_cat=len(cat_dirs)
    images=[]
    labels=[]
    for label,category in enumerate(cat_dirs):
        cat_path=path+category
        img_names=utils.get_dirs(cat_path)
        img_paths=utils.append_path(cat_path,img_names)
        for img_path in img_paths:
            img=read_images(img_path)
            images.append(img)
            labels.append(label)
        images=np.array(images)
        labels=np.array(labels)
    return LabeledImages(images,labels,n_cat)
 
if __name__ == "__main__":
    path="/home/user/cf/conv_frames/cls/images/"
    read_images(path)
