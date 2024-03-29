import numpy as np
import cv2,os.path,math
#from keras.models import load_model
import files,data.seqs

class FrameSeqs(dict):
    def __init__(self, args=[]):
        super(FrameSeqs, self).__init__(args)

    def n_cats(self):
        cats=set([ name_i.get_cat() for name_i in self.keys()])
        return len(cats)

    def n_persons(self):
        persons=set([ name_i.get_person() for name_i in self.keys()])
        return len(persons)

    def n_frames(self):
        n=0
        for seq_i in self.values():
            n+=len(seq_i)
        return n

    def names(self):
        return sorted(self.keys(),key=files.natural_keys) 

    def dims(self):
        shape=self.shape()
        return (shape[0],shape[1])

    def shape(self):
        return list(self.values())[0][0].shape

    def split(self,selector=None):
        train,test=files.split(self,selector)
        return FrameSeqs(train),FrameSeqs(test)

    def scale(self,dims=(64,64),new=False):
        def helper(img_j):
            img_j=cv2.resize(img_j,dsize=dims,interpolation=cv2.INTER_CUBIC)
            if(img_j.ndim==3):
                return img_j
            return np.expand_dims(img_j,axis=-1)
        return self.transform(helper,new,single=True)

    def transform(self,fun,new=False,single=True):
        frame_dict= FrameSeqs() if(new) else self
        for name_i,seq_i in self.items():
            print(name_i)
            if(single):
                frame_dict[name_i]=[fun(img_j)
                            for img_j in seq_i]
            else:
                frame_dict[name_i]=fun(seq_i)

        return frame_dict

    def to_dataset(self):
        names=self.names()
        X=[ np.array(self[name_i]) for name_i in names]
        y=[name_i.get_cat() for name_i in names]
        return np.array(X),y

    def save(self,out_path):
        files.make_dir(out_path)
        for name_i,seq_i in self.items():
            out_i="%s/%s" % (out_path,name_i)
            if( len(self.dims())==3 and self.dims()[-1]!=1):
                seq_i=[np.concatenate(frame_j.T,axis=0) 
                            for frame_j in seq_i]
            save_frames(out_i,seq_i)

    def seqs_len(self):
        return [len(seq_i) for seq_i in self.values()]

    def min_len(self):
        return min(self.seqs_len())

    def subsample(self,size=20):
        if(size):
            fun=MinLength(size)
            self.transform(fun,new=False,single=False)

    def subsample_agum(self,size=20,n_agum=2):
        train,test=self.split()
        fun=MinLength(size)
        test=test.transform(fun,new=False,single=False)
        new_frames={}
        for name_i,frames_i in train.items():
            for j in range(n_agum):
                new_name_i=files.Name("%s_%d" % (name_i,j))
                new_frames[new_name_i]=fun(frames_i)
        full_frames=list(test.items()) + list(new_frames.items())
        return FrameSeqs(full_frames)

class MinLength(object):
    def __init__(self,size):
        self.size = size

    def __call__(self,frames):
        n_frames=len(frames)
        indexes=np.random.randint(n_frames,size=self.size)
        indexes=np.sort(indexes)
        return [frames[i] for i in indexes]

class StaticDownsample(object):
    def __init__(self,size=30):
        self.size=size 

    def __call__(self,frames):
        scale=(len(frames)/self.size)
        indexes=[math.floor(scale*i) 
            for i in range(self.size)]
        return [frames[i] for i in indexes]
#        scale= math.floor(len(frames)/self.size)
#        return [frames[scale*i] for i in range(self.size)]

def read_frame_seqs(in_path,read=None):
    if(read is None):
        read=ReadFrames()
    if(type(read)==int):
        read=ReadFrames(n_split=read)
    frame_seqs=FrameSeqs()
    for i,path_i in enumerate(files.top_files(in_path)):
        name_i=files.Name(path_i.split('/')[-1]).clean()
        if(len(name_i)==0):
            name_i=files.Name(str(i))
        frames=[ read(path_j)#,n_split) 
                for path_j in files.top_files(path_i)]
        frame_seqs[name_i]=frames
    return frame_seqs

class ReadFrames(object):
    def __init__(self,n_split=1,color=cv2.IMREAD_GRAYSCALE):
        color=cv2.IMREAD_GRAYSCALE if(color=="grey") else cv2.IMREAD_COLOR
        self.n_split=n_split
        self.color=color

    def __call__(self,in_path):
        if( os.path.isdir(in_path)):
            return [ self(path_i) 
                for path_i in files.top_files(in_path)]
        n_split=self.n_split
        frame_ij=cv2.imread(in_path,self.color)
        if(n_split is None):
            n_split=int(frame_ij.shape[1] /frame_ij.shape[0])    
        if(n_split==1):
            return frame_ij
        return np.array(np.vsplit(frame_ij,n_split)).T

def save_frames(in_path,frames):
    files.make_dir(in_path)
    for i,frame_i in enumerate(frames):
        out_i="%s/%d.png" % (in_path,i)
        cv2.imwrite(out_i, frame_i)

def rescale_seqs(in_path,out_path,dims=(64,64),n_split=1):
    frame_seqs=read_frame_seqs(in_path,n_split=n_split)
    frame_seqs.scale(dims,new=False)
    frame_seqs.save(out_path)

def tranform_frames(in_path,out_path,fun,whole=False):
    frames=read_frame_seqs(in_path,n_split=1)
    if(whole):
        fun(frames)   
    else:
        frames.transform(fun)
    frames.save(out_path)

def transform_lazy(in_path,out_path,fun,read=None,
            recreate=True,single=False):
    if(read is None):
        read=ReadFrames(color=cv2.IMREAD_COLOR)
    files.make_dir(out_path)
    for i,path_i in enumerate(files.top_files(in_path)):
        name_i=path_i.split("/")[-1]
        out_i="%s/%s" % (out_path,name_i)
        if( recreate or not os.path.exists(out_i)):
            frames=[ read(path_j) 
                for path_j in files.top_files(path_i)]
            if(single):
                frames=[fun(name_i,frame_j) 
                        for frame_j in frames]
            else:
                frames=fun(name_i,frames)
            save_frames(out_i,frames)