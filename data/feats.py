import numpy as np,re
import files

class Feats(dict):
	def __init__(self, arg=[]):
		super(Feats, self).__init__(arg)

	def __add__(self,feat_i):
		names=common_names(self.keys(),feat_i.keys())
		new_feats=Feats()
		for name_i in names:
			x_i=np.concatenate([self[name_i],feat_i[name_i]],axis=0)
			new_feats[name_i]=x_i
		return new_feats

	def n_cats(self):
		return max(self.get_cats())+1

	def dim(self):
		return list(self.values())[0].shape[0]

	def names(self):
		return sorted(self.keys(),key=files.natural_keys) 
	
	def split(self,selector=None):
		train,test=files.split(self,selector)
		return Feats(train),Feats(test)

	def to_dataset(self):
		names=self.names()
		X=np.array([self[name_i] for name_i in names])
		return X,self.get_cats()

	def get_cats(self):
	    return [ int(name_i.split('_')[0])-1 for name_i in self.names()]
	
	def transform(self,extractor):
		feat_dict={	name_i: extractor(feat_i)
				for name_i,feat_i in self.items()}
		return Feats(feat_dict)

	def remove_nan(self):
		for feat_i in self.values():
			feat_i[np.isnan(feat_i)]=0

	def has_nan(self):
		return any([np.isnan(x_i).any() for x_i in self.values()])

	def norm(self):
		X,y=self.to_dataset()
		mean=np.mean(X,axis=0)
		std=np.std(X,axis=0)
		std[np.isnan(std)]=1
		std[std==0]=1
		for name_i,feat_i in self.items():
			self[name_i]=(feat_i-mean)/std

	def save(self,out_path):
		lines=[]
		for name_i,feat_i in self.items():
			txt_i=np.array2string(feat_i,separator=",")
			txt_i=txt_i.replace("\n","")
			lines.append("%s#%s" % (txt_i,name_i))
		feat_txt='\n'.join(lines)
		feat_txt=feat_txt.replace('[','').replace(']','')
		feat_txt = feat_txt.replace(' ','')
		file_str = open(out_path,'w')
		file_str.write(feat_txt)
		file_str.close()

def read_feats(in_path):
    if(type(in_path)==list):
        all_feats=[read_feats(path_i) for path_i in in_path]
        return concat_feats(all_feats)
    lines=open(in_path,'r').readlines()
    feat_dict=Feats()
    for line_i in lines:
        raw=line_i.split('#')
        if(len(raw)>1):
            data_i,info_i=raw[0],raw[-1]
            info_i= files.Name(info_i).clean()
            feat_dict[info_i]=np.fromstring(data_i,sep=',')
    return feat_dict

def concat_feats(all_feats):
	first=all_feats[0]
	for feat_i in all_feats[1:]:
		first+=feat_i
	return first

def common_names(names1,names2):
	return list(set(names1).intersection(set(names2)))

def unified_exp(in_path):
	all_feats=read_feats(files.top_files(in_path))
	train_model(all_feats)

def get_feats(in_path,fun):
    feats=Feats()
    for i,path_i in enumerate(files.top_files(in_path)):
        name_i=files.get_name(path_i)
        feats[name_i]=fun(path_i)
    return feats