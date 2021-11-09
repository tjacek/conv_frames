import numpy as np
import cv2
import files,data.imgs

class ActionImgs(dict):
	def __init__(self, arg=[]):
		super(ActionImgs, self).__init__(arg)

	def dims(self):
		return list(self.values())[0].shape
    
	def scale(self,dims=(64,64)):
		def helper(img_i):
			return scale_frames(img_i,dims) 
		self.transform(helper)

	def add_dim(self):
		self.transform(lambda img_i: np.expand_dims(img_i,axis=-1))
	
	def transform(self,fun,copy=False):
		data_dict=ActionImgs() if(copy) else self
		for name_i,img_i in self.items():
			data_dict[name_i]=fun(img_i)
		return data_dict

	def names(self):
		return sorted(self.keys(),key=files.natural_keys) 

	def split(self,selector=None):
		train,test=files.split(self,selector)
		return ActionImgs(train),ActionImgs(test)

	def save(self,out_path):
		files.make_dir(out_path)
		for name_i,img_i in self.items():
			out_i="%s/%s.png" % (out_path,name_i)
			cv2.imwrite(out_i, img_i)

def scale_frames(frames,dims):
    if(type(frames)==list):
    	return [ scale_frames(frame_i,dims) for frame_i in frames]
    return cv2.resize(frames,dsize=dims,
					interpolation=cv2.INTER_CUBIC)

def read_actions(in_path,img_type="grey"):
	color= cv2.IMREAD_GRAYSCALE if(img_type=="grey") else  cv2.IMREAD_COLOR
	actions=ActionImgs()
	for path_i in files.top_files(in_path):
		name_i=files.Name(path_i.split('/')[-1])
		name_i=name_i.clean()
		img_i=cv2.imread(path_i,color)
		if(img_i is None):
			raise Exception(path_i)
		actions[name_i]=img_i
	return actions

def get_actions(in_path,fun,out_path=None,dims=(64,64)):
	frame_seqs=data.imgs.read_frame_seqs(in_path)#,n_split=1)
	actions=ActionImgs()
	for name_i,seq_i in frame_seqs.items():
		print(name_i)
		actions[name_i]=fun(seq_i)
	if(dims):
		actions.scale(dims)
	if(out_path):
		actions.save(out_path)
	return actions

def get_actions_eff(in_path,fun,out_path=None,dims=None):
    read=data.imgs.ReadFrames(1,cv2.IMREAD_COLOR)
    files.make_dir(out_path)
    for i,path_i in enumerate(files.top_files(in_path)):
        name_i=files.Name(path_i.split('/')[-1]).clean()
        frames=[ read(path_j)#,n_split) 
                for path_j in files.top_files(path_i)]
        frames=[ frame_i for frame_i in frames
                    if(not (frame_i is None))]
        print(i)
        action_i=fun(frames)
        out_i="%s/%s.png" % (out_path,name_i)
        cv2.imwrite(out_i, action_i)

def tranform_actions(in_path,out_path,fun):
	actions=read_actions(in_path)
	actions.transform(fun)
	actions.save(out_path)

def transform_lazy(in_path,out_path,fun):
    color=cv2.IMREAD_COLOR
    files.make_dir(out_path)
    for i,path_i in enumerate(files.top_files(in_path)):
        action_i=cv2.imread(path_i,color)
        name_i=path_i.split("/")[-1]
        action_i=fun(name_i,action_i)
        out_i="%s/%s" % (out_path,name_i)
        print(action_i.shape)
        cv2.imwrite(out_i, action_i)

def from_paths(paths,color=cv2.IMREAD_GRAYSCALE):
    action_imgs= ActionImgs()
    for i,path_i in enumerate(paths):
        print(path_i)
        action_i=cv2.imread(path_i,color)
        name_i=files.get_name(path_i)
        action_imgs[name_i]=action_i
    return action_imgs