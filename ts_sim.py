from keras.layers import Input,Dense
from keras.models import Model
import keras
import data.seqs,deep

class TS_SIM(object):
    def __init__(self, n_hidden=100,activ='relu',l1=0.01):
        self.activ=activ
        self.n_hidden=n_hidden
        self.l1=l1
        self.n_kerns=[32,32]
        self.kern_size=[8,8]
        self.pool_size=[2,2]
        self.dropout=0.5

    def __call__(self,params):
        input_img=Input(shape=(params['dims']))
        x=deep.add_conv_layer(input_img,self.n_kerns,self.kern_size,
                            self.pool_size,activ=self.activ,one_dim=True)
        x=deep.full_layer(x,size=100,l1=self.l1,
            dropout=self.dropout,activ=self.activ)
        x=Dense(units=params['n_cats'],activation='softmax')(x)
        model = Model(input_img, x)
        model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam')
        model.summary()
        return model

def train_sim(in_path):
    seq_dict=data.seqs.read_seqs(in_path)
    train=seq_dict.split()[0]
    X,y=train.to_dataset()
    params={"dims":train.dims(),"n_cats":max(y)+1}
    make_nn=TS_SIM()
    model=make_nn(params)
    print(params)

train_sim("seqs")