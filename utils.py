import numpy as np
from collections import defaultdict
import data.feats

def residuals(true_path,pred_path,out_path):
    true_feats=data.feats.read_feats(true_path)
    pred_feats=data.feats.read_feats(pred_path)
    name_id=name_map(pred_feats)
    res_feats=data.feats.Feats()
    for name_i,name_list_i in name_id.items():
        for name_j in name_list_i:
            res_feats[name_j]= true_feats[name_i]-pred_feats[name_j]
    res_feats.save(out_path)

def name_map(res_feats):
	name_id=defaultdict(lambda :[])
	for name_i in res_feats.keys():
		name_id[name_i.subname(3)].append(name_i)
	return name_id

def fill_agum( raw_feats,agum_feats,data_type=data.feats.Feats):
    name_id=name_map(agum_feats)
    full_feats=data.feats.Feats()
    for name_i,name_list_i in name_id.items():
        for name_j in name_list_i:
            full_feats[name_j]=raw_feats[name_i]
    return full_feats

def validate_regresion(true_path,pred_path):
    true_feats=data.feats.read_feats(true_path).split()[1]
    pred_feats=data.feats.read_feats(pred_path).split()[1]
    error=[ (true_feats[name_i]-pred_feats[name_i])**2  
                for name_i in true_feats.keys()]
    mse=np.mean(error)
    print(mse)

if __name__ == "__main__":
    true_path="smooth3/simple/feats"
    pred_path="smooth3/simple/base_120/feats"
    validate_regresion(true_path,pred_path)