import files#,utils,spline

class EnsTransform(object):
	def __init__(self,funcs,dir_names,input_dir="seqs"):
		self.funcs=funcs
		self.dir_names=dir_names
		self.input_dir=input_dir

	def __call__(self,input_paths,out_path, arg_dict):
		dirs=files.get_paths(out_path,self.dir_names)
		for dir_i in dirs.values():
			files.make_dir(dir_i)
		for path_i in input_paths:
			name_i=path_i.split('/')[-1]
			args_i={ key_i:"%s/%s" % (path_i,name_i) 
				for key_i,path_i in dirs.items()}
			args_i={**args_i,**arg_dict}
			args_i[self.input_dir]=path_i
			for fun,arg_names in self.funcs:
				fun_args=[args_i[name_k]  
							for name_k in arg_names]
				fun(*fun_args)

class BinaryEns(object):
    def __init__(self,read,train,extract):
        self.read=read
        self.train=train
        self.extract=extract

    def __call__(self,in_path,ens_path,n_epochs=5):
        frame_seqs=self.read(in_path)
        files.make_dir(ens_path)
        files.make_dir("%s/nn" % ens_path)
        files.make_dir("%s/feats" % ens_path)
        n_cats=frame_seqs.n_cats()
        for i in range(n_cats):
            nn_i="%s/nn/%d" % (ens_path,i)
            feats_i="%s/feats/%d" % (ens_path,i)
            frame_i=binarize_dict(frame_seqs,i)
#            self.train(frame_i,nn_i,n_epochs=n_epochs)
            self.extract(frame_seqs,nn_i,feats_i)

def binarize_dict(frame_seqs,cat):
    dict_type=type(frame_seqs)
    binary_dict=dict_type()
    for i,name_i in enumerate(frame_seqs.keys()):
        new_name=binarize_name(name_i,i,cat+1)
        binary_dict[new_name]=frame_seqs[name_i]
    return binary_dict

def binarize_name(name_i,i,cat):
    raw=name_i.split("_")
    binary=int(int(raw[0])==cat)
    new_name="%d_%s_%d" % (binary,raw[1],i)
    return files.Name(new_name)
    
#def binarize(y,i):
#	y_i=[ int(y_j==i) for y_j in y]
#	return utils.to_one_hot(y_i,n_cats=2)