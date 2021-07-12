import tensorflow.keras
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import data.feats

def make_nn(params):
    model = Sequential()
    model.add(Dense(100, input_dim=params['dims'], activation='relu'))
    model.add(Dense(params['n_cats'], activation='relu'))
    model.compile('adam','binary_crossentropy', metrics=['accuracy'])
    model.summary()

def simple_exp(in_path):
    feat_dict=data.feats.read_feats(in_path)
    X,y=feat_dict.to_dataset()
    params={'dims':X.shape[1],'n_cats':max(y)+1}
    make_nn(params)	

in_path="../../2021_VI/ICCCI/ens_splitI/feats/0"
simple_exp(in_path)