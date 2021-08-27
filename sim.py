from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization


def make_sim(params):
    model = Sequential()
    model.add(Dense(100, input_dim=params['dims'], activation='relu',name="hidden",
        kernel_regularizer=regularizers.l1(0.001)))
    model.add(Dense(64, activation='relu'))

def single_exp(feat_path,nn_path,out_path,n_epochs=5):
    read=data.feats.read_feats