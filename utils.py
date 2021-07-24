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

true_path="../3DHOI/1D_CNN/feats"
pred_path="student/ae_30_200_agum/feats"
out_path="student/ae_30_200_agum/res"
residuals(true_path,pred_path,out_path)