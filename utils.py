import data.feats

def residuals(true_path,pred_path,out_path):
    true_feats=data.feats.read_feats(true_path)
    pred_feats=data.feats.read_feats(pred_path)
    res_feats=data.feats.Feats()
    for name_i in true_feats.keys():
    	res_feats[name_i]= true_feats[name_i]-pred_feats[name_i]
    res_feats.save(out_path)

true_path="../3DHOI/1D_CNN/feats"
pred_path="student/ae_30_200/feats"
out_path="student/ae_30_200/res"
residuals(true_path,pred_path,out_path)