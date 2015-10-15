import imp
utils =imp.load_source("utils","/home/user/cf/conv_frames/utils.py")
import seq

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
    vectors,bow=create_vectors(dataset)
    vectors=standarize_vectors(vectors,bow)
    return vectors

def create_vectors(dataset):
    bow=BOW()
    vectors=[]
    for instance in dataset.instances:
        sub_seqs=instance.sub_seq()
        sub_seqs=[bow.get_index(word) for word in sub_seqs]
        vector=utils.get_zeros(bow.size)
        for sub_seq in sub_seqs:
            vector[sub_seq]+=1
        vectors.append(vector)
    return vectors,bow

def standarize_vectors(vectors,bow):
    def standarize(vec):
        diff=bow.size-len(vec)
        if(diff>0):
            vec+=utils.get_zeros(diff)
        norm_const=float(sum(vec))
        vec=[float(cord_i)/norm_const for cord_i in vec]
        return vec
    return map(standarize,vectors)

if __name__ == "__main__":
    in_path="/home/user/cf/seqs/dataset.seq"
    extract_bow(in_path)
