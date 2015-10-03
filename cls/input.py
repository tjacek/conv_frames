import imp
utils =imp.load_source("utils","/home/user/cf/conv_frames/utils.py")
import numpy as np

class LabeledImages(object):
    def __init__(self,images,labels,n_cat):
        self.X=images
        self.y=labels
        self.n_cat=n_cat
        self.size=self.X.shape[0]
        self.image_shape=self.X[0].shape

    #def get_image_shape(self):
    #    return self.X[0].shape

def load_images(path):
    cat_dirs=utils.get_dirs(path)
    n_cat=len(cat_dirs)
    images=[]
    labels=[]
    for label,category in enumerate(cat_dirs):
        cat_path=path+category
        img_names=utils.get_paths(cat_path)
        for img in utils.read_images(img_names):
            images.append(img)
            labels.append(label)
    images=np.array(images)
    labels=np.array(labels)
    return LabeledImages(images,labels,n_cat)
 
if __name__ == "__main__":
    path="/home/user/cf/conv_frames/cls/images/"
    images=load_images(path)
    print(images.image_shape)
