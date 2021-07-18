import numpy as np
import data.feats
import tc_nn,learn,files

def train_student(frame_path,teacher_path,out_path,n_epochs=100):
    files.make_dir(out_path)
    frame_seq,teacher_feat,params=read_data(frame_path,teacher_path)
#    flip_agum(frame_seq,teacher_feat)
    make_tcn=tc_nn.TC_NN(n_hidden=100,batch=True,loss='mean_squared_error')
    model=make_tcn(params)
    y=teacher_feat.to_dataset()[0]
    X=frame_seq.to_dataset()[0]
    model.fit(X,y,epochs=n_epochs,batch_size=16)
    tc_nn.save(model,"%s/nn" % out_path)
    return params["n_cats"]

def read_data(frame_path,teacher_path):
    read=tc_nn.get_read(seq_len=20,dim=(64,64))
    frame_seq=read(frame_path)
    frame_seq=frame_seq.split()[0]
    teacher_feat=data.feats.read_feats(teacher_path)
    teacher_feat=teacher_feat.split()[0]
    params={'seq_len':frame_seq.min_len(),'dims':frame_seq.dims(),
                'n_cats': teacher_feat.dim() }#frame_seq.n_cats() * frame_seq.n_cats()}
    return frame_seq,teacher_feat,params

def extract_student(frame_path,nn_path,out_path,n_cats):
    read=tc_nn.get_read(seq_len=20,dim=(64,64))
    make_tcn=tc_nn.TC_NN(loss='mean_squared_error')
    frame_seq=read(frame_path)
    params={'seq_len':frame_seq.min_len(),'dims':frame_seq.dims(),
                'n_cats':n_cats}
    model=tc_nn.read_model(frame_seq,"%s/nn" %nn_path,make_tcn,params)
    model.summary()
    X,y=frame_seq.to_dataset()
    feats=learn.get_features(frame_seq,model)
    feats.save("%s/feats"  % out_path)

def flip_agum(frame_seqs,teacher_feat):
    pairs=list(frame_seqs.items())
    for name_i,seq_i in pairs:
        new_seq_i=[np.fliplr(frame_j) for frame_j in seq_i] 
        new_name_i=files.Name("%s_1" % name_i)
        frame_seqs[new_name_i]=new_seq_i
        teacher_feat[new_name_i]=teacher_feat[name_i]
    return frame_seqs,teacher_feat

frame_path="../3DHOI/frames"
teacher_path="../ml_utils/3DHOI_simple"
nn_path="student_simple"
n_cats=train_student(frame_path,teacher_path,nn_path,n_epochs=5)
extract_student(frame_path,nn_path,nn_path,n_cats)