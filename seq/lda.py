import seq
import simple_bow as bow
import numpy as np
from sklearn.lda import LDA
import imp
utils =imp.load_source("utils","/home/user/cf/conv_frames/utils.py")

def lda_features(train_path,test_path):
    train,test=bow.to_bow(train_path,test_path)
    train,lda_cls=create_lda(train)
    test=apply_lda(test,lda_cls)
    save_lda(train,train_path)
    save_lda(test,test_path)

def basic_bow(in_path,_bow=None):
    dataset=seq.create_dataset(in_path)
    labels=dataset.get_labels()
    if(_bow==None):
        vectors,_bow=bow.compute_bow(dataset,True)
        vectors=np.array(vectors)
        return _bow,(vectors,labels)
    else:
        vectors=bow.apply_bow(dataset,_bow)
        vectors=np.array(vectors)
        return vectors,labels

def save_lda(dataset,out_path):
    out_path=utils.change_postfix(out_path)
    utils.to_labeled_file(out_path,dataset[0],dataset[1])

def create_lda(dataset):
    x,y=dataset
    clf = LDA(n_components=13)
    clf.fit(x,y)
    x_t=clf.transform(x)
    return (x_t,y),clf

def apply_lda(dataset,lda_cls):
    x,y=dataset
    x_t=lda_cls.transform(x)
    return (x_t,y)

if __name__ == "__main__":
   path="/home/user/cf/seqs/"
   train_path=path+"dataset_train.seq"
   test_path=path+"dataset_test.seq"
   lda_features(train_path,test_path)
