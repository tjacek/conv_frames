import tensorflow.keras
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import data.feats,learn

def make_nn(params):
    model = Sequential()
    model.add(Dense(100, input_dim=params['dims'], activation='relu'))
    model.add(Dense(params['n_cats'], activation='relu'))
    model.compile('adam','binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def simple_exp(feat_path,out_path,n_epochs=5):
    read=data.feats.read_feats#(in_path)
    train=learn.Train(to_dataset,make_nn,read,batch_size=16)
    train(feat_path,out_path,n_epochs)	

def to_dataset(feat_dict):
    X,y=feat_dict.to_dataset()
    params={'dims':X.shape[1],'n_cats':max(y)+1}
    return X,y,params

in_path="../../2021_VI/ICCCI/ens_splitI/feats/0"
simple_exp(in_path,"test")