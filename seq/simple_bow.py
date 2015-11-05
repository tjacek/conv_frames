import imp
utils =imp.load_source("utils","/home/user/cf/conv_frames/utils.py")
import seq
import numpy as np

class BOW(object):
    def __init__(self):
        self.indexes={}
        self.size=0
    
    def get_index(self,word):
        if(not word in self.indexes):
            self.indexes[word]=self.size
            self.size+=1
        return self.indexes[word]

def extract_bow(in_path):
    get_labeled_vectors(in_path,compute_bow)

def get_labeled_vectors(in_path,compute_features,suffix=".lb"):
    out_path=in_path.replace(".seq",suffix)
    dataset=seq.create_dataset(in_path)
    labels=dataset.get_labels()
    vectors=compute_features(dataset)
    utils.to_labeled_file(out_path,vectors,labels)

def get_unlabeled_vectors(in_path,compute_features,suffix=".csv"):
    out_path=in_path.replace(".seq",suffix)
    dataset=seq.create_dataset(in_path)
    vectors=compute_features(dataset)
    utils.to_csv_file(out_path,vectors,labels)

def compute_bow(dataset):
    bow=create_dict(dataset.instances)
    vectors=create_vectors(dataset,bow)
    return vectors

def create_dict(instances):
    bow=BOW()
    for instance in instances:
        sub_seqs=instance.sub_seq()
        sub_seqs=[bow.get_index(word) for word in sub_seqs]
    return bow

def create_vectors(dataset,bow):
    vectors=[]
    for instance in dataset.instances:
        sub_seqs=instance.sub_seq()
        sub_seqs=[bow.get_index(word) for word in sub_seqs]
        vector=utils.get_zeros(bow.size)
        for sub_seq in sub_seqs:
            vector[sub_seq]+=1
        norm_const=float(sum(vector))
        vector=[float(cord_i)/norm_const for cord_i in vector]
        vectors.append(vector)
    return td_idf(vectors)

def get_idf(vectors):
    d=float(len(vectors))
    size=len(vectors[0])
    vectors=np.array(vectors)
    idf=[map(binarize,vectors[:,i])  for i in range(size)]
    idf=[sum(vec) for vec in idf]
    idf=[np.log(d/ dt) for dt in idf]
    return idf,vectors

def td_idf(vectors):
    idf,vectors=get_idf(vectors)
    new_vectors=[idf[i]*vectors[:,i] for i in range(len(idf))]
    return np.array(new_vectors)    

def binarize(x):
    if(x==0):
        return 0.0
    else:
        return 1.0

def splited_bow(train_path,test_path):
    train,test=to_bow(train_path,test_path)
    train_path=utils.change_postfix(train_path)
    utils.to_labeled_file(train_path,train[0],train[1])
    test_path=utils.change_postfix(test_path)
    utils.to_labeled_file(test_path,test[0],test[1])

def to_bow(train_path,test_path):
    train=seq.create_dataset(train_path)
    test=seq.create_dataset(test_path)
    bow=create_dict(train.instances +test.instances)
    train_feats=create_vectors(train,bow)
    test_feats=create_vectors(test,bow)
    train_data=(train_feats,train.get_labels())
    test_data=(test_feats,test.get_labels())
    return train_data,test_data

if __name__ == "__main__":
    path="/home/user/cf/seqs/dataset"
    train_path=path+"_train.seq"
    test_path=path+"_test.seq"
    splited_bow(train_path,test_path)
   #in_path="/home/user/cf/seqs/dataset_test.seq"
   #extract_bow(in_path)
