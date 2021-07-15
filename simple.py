import tensorflow.keras
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import data.feats,learn,ens,files

def make_nn(params):
    model = Sequential()
    model.add(Dense(100, input_dim=params['dims'], activation='relu',name="hidden"))
    model.add(Dense(params['n_cats'], activation='relu'))
    model.compile('adam','binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def ensemble_exp(in_path,out_path,n_epochs=5):
    input_paths=files.top_files(in_path)
    print(input_paths)
    read=data.feats.read_feats
    train=learn.Train(to_dataset,make_nn,read,batch_size=16)
    extract=learn.Extract(make_nn,read,name="hidden")
    funcs=[[train,["feats","nn","n_epochs"]],
           [extract,["feats","nn","simple_feats"]]]
    dir_names=["nn","simple_feats"]
    ensemble=ens.EnsTransform(funcs,dir_names,"feats")
    files.make_dir(out_path)
    ensemble(input_paths,out_path, arg_dict={"n_epochs":n_epochs})

def single_exp(feat_path,nn_path,out_path,n_epochs=5):
    read=data.feats.read_feats#(in_path)
    train=learn.Train(to_dataset,make_nn,read,batch_size=16)
    train(feat_path,nn_path,n_epochs)
    extract=learn.Extract(make_nn,read=read,name="hidden")
    extract(feat_path,nn_path,out_path)

def to_dataset(feat_dict):
    X,y=feat_dict.to_dataset()
    params={'dims':X.shape[1],'n_cats':max(y)+1}
    return X,y,params

in_path="../../2021_VI/ICCCI/ens_splitI/feats"
#single_exp(in_path,"nn","feat",n_epochs=5)
ensemble_exp(in_path,"test",n_epochs=5)