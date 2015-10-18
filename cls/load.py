import imp
utils =imp.load_source("utils","/home/user/cf/conv_frames/utils.py")
import numpy as np

class LabeledImages(object):
    def __init__(self,images,labels,n_cats):
        self.X=images
        self.y=labels
        self.n_cats=n_cats
        self.n_images=self.X.shape[0]
        self.image_shape=self.X[0].shape
        self.image_size=self.image_shape[0]*self.image_shape[1]

    def shape(self):
        return (self.image_size,self.n_cats)

    def get_flatten_images(self):
        flat_images=[x_i.flatten() for x_i in self.X]
        return np.array(flat_images)

    def get_number_of_batches(self,batch_size):
        n_batches=self.n_images / batch_size
        if(self.n_images % batch_size != 0):
            n_batches+=1
        return n_batches

    def get_batches(self,batch_size,flat=True):
        n_batches=self.get_number_of_batches(batch_size)
        iter_batches=range(n_batches)
        if(flat):
            x_full=self.get_flatten_images()
        else:
            x_full=self.X
        X_b=[get_batch(i,x_full,batch_size) for i in iter_batches]
        X_b=np.array(X_b)
        y_b=[get_batch(i,self.y,batch_size) for i in iter_batches]
        y_b=np.array(y_b)
        return X_b,y_b

    def single_batch(self,flat=True):
        single_batch_size=len(self.y)
        return self.get_batches(single_batch_size,flat)

def get_batch(i,full_data,batch_size):
    return full_data[i * batch_size: (i+1) * batch_size]

def get_images(path):
    cat_dirs=utils.get_dirs(path)
    print(cat_dirs)
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
    images=images.astype(float)
    labels=np.array(labels)
    return LabeledImages(images,labels,n_cat)
 
if __name__ == "__main__":
    path="/home/user/cf/conv_frames/cls/images/"
    images=get_images(path)
    print(images.image_shape)
