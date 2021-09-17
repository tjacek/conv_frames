import numpy as np
from tensorflow.keras.layers import Input,Dense,Flatten
from tensorflow.keras.models import Model
import data.actions,data.imgs
import sim_core,deep,learn

class FrameSim(object):
    def __init__(self,n_hidden=128):
        self.n_hidden=n_hidden
        self.n_kerns=[64,32]
        self.kern_size=[(5,5),(3,3),(3,3)]
        self.pool_size=[(4,4),(3,3),(2,2)]  

    def __call__(self,params):
        input_shape=params["input_shape"]
        model,feature_extractor=sim_core.sim_template(input_shape,self)
        model.compile(loss=sim_core.contrastive_loss, optimizer="adam")
        feature_extractor.summary()
        return model,feature_extractor

    def build_model(self,input_shape):
        inputs = Input(input_shape)
        x=deep.add_conv_layer(inputs,self.n_kerns,self.kern_size,
                self.pool_size,one_dim=False)
        x=Flatten()(x)
        x=Dense(self.n_hidden, activation='relu',
            name='hidden',kernel_regularizer=None)(x)
        model = Model(inputs, x)
        return model

def train(in_path,out_path,n_epochs=5,dims=(128,64)):
    make_nn=FrameSim()
    def read(in_path):
        actions=data.actions.get_actions(in_path,
            center_frame,out_path=None,dims=dims)
#        actions.transform(lambda img_i: np.expand_dims(img_i,axis=-1))
#        raise Exception( actions.dim())
        return actions
    train_sim=learn.SimTrain(read,make_nn,to_dataset,n_batch=8)
    train_sim(in_path,out_path,n_epochs)
#    train,test=action_dict.split()
#    X,y=sim_core.pair_dataset(train)
#    X=[ np.expand_dims(x_i,axis=-1) for x_i in X]
#    params={"input_shape":(*train.dim(),1)}
#    make_nn=FrameSim()
#    model,extractor=make_nn(params)
#    model.fit(X,y,epochs=n_epochs,batch_size=batch_size)
#    if(out_path):
#        extractor.save(out_path)

def to_dataset(train):
    X,y=sim_core.pair_dataset(train)
    params={ "input_shape":(*train.dim(),1)}
    X=[ np.expand_dims(x_i,axis=-1) for x_i in X]
    return X,y,params

def extract(in_path,nn_path,out_path):
    import tc_nn
    read=tc_nn.SimpleRead(dim=(64,128),preproc=data.imgs.Downsample())
    frame_dict= read(in_path)
    model=learn.base_read_model(frame_dict,nn_path)
    extractor=learn.get_extractor(model,"hidden")
    def helper(img_i):
        img_i=np.array(img_i)
        img_i=np.swapaxes(img_i, 1, 2)
        feat_i=extractor.predict(img_i)
        return feat_i
    seq_dict=frame_dict.transform(helper,new=True,single=False)
    seq_dict=data.seqs.Seqs(seq_dict)
    seq_dict.save(out_path)

def center_frame(frames):
    center= int(len(frames)/2)
    return frames[center]

in_path="../best2/frames"
nn_path="../deep_dtw/nn"
out_path="../deep_dtw/seqs"
train(in_path,nn_path)
#extract(in_path,nn_path,out_path)