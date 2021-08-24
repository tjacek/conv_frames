import tensorflow.keras
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization
#from keras.layers.normalization import BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
import data.feats,learn,ens,files

def make_nn(params):
    model = Sequential()
    model.add(Dense(100, input_dim=params['dims'], activation='relu',name="hidden",
        kernel_regularizer=regularizers.l1(0.001)))
    model.add(BatchNormalization())
    model.add(Dense(params['n_cats'], activation='softmax'))
    optim=optimizers.RMSprop(learning_rate=0.00001)
    model.compile(loss='categorical_crossentropy',optimizer=optim, metrics=['accuracy'])
    model.summary()
    return model

def ensemble_exp(common,binary,out_path,n_epochs=5):
    input_paths=files.top_files(binary)
    print(input_paths)
    def read(feat_path):
        feats_i=data.feats.read_feats(common+[feat_path])
        feats_i.norm()
        feats_i.remove_nan()
        return feats_i
    train=learn.Train(to_dataset,make_nn,read,batch_size=16)
    extract=learn.Extract(make_nn,read,name="hidden")
    funcs=[[train,["feats","nn","n_epochs"]],
           [extract,["feats","nn","simple_feats"]]]
    dir_names=["nn","simple_feats"]
    ensemble=ens.EnsTransform(funcs,dir_names,"feats")
    files.make_dir(out_path)
    ensemble(input_paths,out_path, arg_dict={"n_epochs":n_epochs})

def single_exp(feat_path,nn_path,out_path,n_epochs=5):
    read=data.feats.read_feats
    train=learn.Train(to_dataset,make_nn,read,batch_size=16)
    train(feat_path,nn_path,n_epochs)
    extract=learn.Extract(make_nn,read=read,name="hidden")
    extract(feat_path,nn_path,out_path)

def to_dataset(feat_dict):
    X,y=feat_dict.to_dataset()
    params={'dims':X.shape[1],'n_cats':max(y)+1}
    return X,y,params

in_path="../ICCCI/3DHOI"
common=["1D_CNN/feats","dtw/corl/dtw","dtw/max_z/dtw"]
binary="ens_splitI/feats" 
paths=files.ens_paths(in_path,common,binary)
#single_exp("test","nn","feat",n_epochs=5)
ensemble_exp(paths["common"],paths["binary"],"simple",n_epochs=100)
