import numpy as np
import data.feats
import tc_nn,learn,files,ens

class TrainStudent(object):
    def __init__(self,make_tcn=None,batch_size=16):
        if(make_tcn is None):
            make_tcn=tc_nn.TC_NN(n_hidden=100,batch=True,loss='mean_squared_error')
        self.make_tcn=make_tcn
        self.batch_size=batch_size

    def __call__(self,frame_path,teacher_path,out_path,n_epochs=5):
        files.make_dir(out_path)
        frame_seq,teacher_feat,params=read_data(frame_path,teacher_path)
        model=self.make_tcn(params)
        y=teacher_feat.to_dataset()[0]
        X=frame_seq.to_dataset()[0]
        model.fit(X,y,epochs=n_epochs,batch_size=self.batch_size)
        tc_nn.save(model,"%s/nn" % out_path)
        return params["n_cats"]

class ExtractStudent(object):
    def __init__(self, make_tcn=None):
        if(make_tcn is None):
            make_tcn=tc_nn.TC_NN(loss='mean_squared_error')
        self.make_tcn=make_tcn
        self.read=tc_nn.get_read(seq_len=20,dim=(64,64))

    def __call__(self,frame_path,nn_path,out_path,n_cats):
        frame_seq=self.read(frame_path)
        params={'seq_len':frame_seq.min_len(),'dims':frame_seq.dims(),
                'n_cats':n_cats}
        model=tc_nn.read_model(frame_seq,"%s/nn" %nn_path,self.make_tcn,params)
        model.summary()
        X,y=frame_seq.to_dataset()
        feats=learn.get_features(frame_seq,model)
        feats.save(out_path)#"%s/feats"  % out_path)

def read_data(frame_path,teacher_path):
    read=tc_nn.get_read(seq_len=20,dim=(64,64))
    frame_seq=read(frame_path)
    frame_seq=frame_seq.split()[0]
    teacher_feat=data.feats.read_feats(teacher_path)
    teacher_feat=teacher_feat.split()[0]
    params={'seq_len':frame_seq.min_len(),'dims':frame_seq.dims(),
                'n_cats': teacher_feat.dim() }
    return frame_seq,teacher_feat,params

def flip_agum(frame_seqs,teacher_feat):
    pairs=list(frame_seqs.items())
    for name_i,seq_i in pairs:
        new_seq_i=[np.fliplr(frame_j) for frame_j in seq_i] 
        new_name_i=files.Name("%s_1" % name_i)
        frame_seqs[new_name_i]=new_seq_i
        teacher_feat[new_name_i]=teacher_feat[name_i]
    return frame_seqs,teacher_feat

def single_student(frame_path,teacher_path,nn_path,n_epochs=100):
    make_tcn=tc_nn.TC_NN(n_hidden=100,batch=False,loss='mean_squared_error')
    train,extract=TrainStudent(make_tcn),ExtractStudent(make_tcn)
    n_cats=train(frame_path,teacher_path,nn_path,n_epochs=100)
    np.set_printoptions(threshold=n_cats)
    extract(frame_path,nn_path,nn_path,n_cats)

def ens_student(frame_path,student_path,out_path,n_epochs=5):
    make_tcn=tc_nn.TC_NN(n_hidden=100,batch=True,loss='mean_squared_error')
    train,extract=TrainStudent(make_tcn),ExtractStudent(make_tcn)
    funcs=[[train,[ "frame","teacher","nn","n_epochs"]],
           [extract,[ "frame","nn","feats","n_cats"]]]
    dir_names=["nn","feats"]
    ensemble=ens.EnsTransform(funcs,dir_names,"teacher")
    files.make_dir(out_path)
    input_paths=files.top_files(student_path)
    args={"n_epochs":n_epochs,"n_cats":100,"frame":frame_path}
    ensemble(input_paths,out_path, arg_dict=args)

frame_path="../3DHOI/frames"
teacher_path="../ml_utils/3DHOI_simple"
nn_path="student_simple_nobatch"
#single_student(frame_path,teacher_path,nn_path,n_epochs=100)
ens_student(frame_path,"test/simple_feats","student_ens",n_epochs=100)