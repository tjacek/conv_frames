import data.feats
import tc_nn,learn,files

def train_student(frame_path,teacher_path,out_path,n_epochs=100):
    files.make_dir(out_path)
    read=tc_nn.get_read(seq_len=20,dim=(64,64))
    frame_seq=read(frame_path)
    frame_seq=frame_seq.split()[0]
    make_tcn=tc_nn.TC_NN(loss='mean_squared_error')
    params={'seq_len':frame_seq.min_len(),'dims':frame_seq.dims(),
                'n_cats':frame_seq.n_cats() * frame_seq.n_cats()}
    model=make_tcn(params)
    teacher_feat=data.feats.read_feats(teacher_path)
    teacher_feat=teacher_feat.split()[0]
    y=teacher_feat.to_dataset()[0]
    X=frame_seq.to_dataset()[0]
    model.fit(X,y,epochs=n_epochs)
    learn.save(model,"%s/nn" % out_path)

def extract_student(frame_path,nn_path,out_path):
    read=tc_nn.get_read(seq_len=20,dim=(64,64))
    make_tcn=tc_nn.TC_NN(loss='mean_squared_error')
    frame_seq=read(frame_path)
    params={'seq_len':frame_seq.min_len(),'dims':frame_seq.dims(),
                'n_cats':frame_seq.n_cats()*frame_seq.n_cats()}
    model=learn.read_model(frame_seq,"%s/nn" %nn_path,make_tcn,params)
    model.summary()
    X,y=frame_seq.to_dataset()
    feats=learn.get_features(frame_seq,model)
    feats.save("%s/feats"  % out_path)

frame_path="../../2021_VI/raw_3DHOI/3DHOI/frames"
teacher_path="../ml_utils/3DHOI"
nn_path="student"
train_student(frame_path,teacher_path,"student")
extract_student(frame_path,nn_path,nn_path)