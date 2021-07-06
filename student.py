import data.feats
import tc_nn,learn,files

def train_student(frame_path,teacher_path,out_path,n_epochs=5):
    files.make_dir(out_path)
    read=tc_nn.get_read(seq_len=20,dim=(64,64))
    frame_seq=read(frame_path)
    make_tcn=tc_nn.TC_NN(loss='mean_squared_error')
    params={'seq_len':frame_seq.min_len(),'dims':frame_seq.dims(),
                'n_cats':frame_seq.n_cats() * frame_seq.n_cats()}
    model=make_tcn(params)
    teacher_feat=data.feats.read_feats(teacher_path)
    y=teacher_feat.to_dataset()[0]
    X=frame_seq.to_dataset()[0]
    model.fit(X,y,epochs=n_epochs)
    learn.save(model,"%s/nn" % out_path)

frame_path="../../2021_VI/raw_3DHOI/3DHOI/frames"
teacher_path="../ml_utils/3DHOI"
train_student(frame_path,teacher_path,"student")