#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense,BatchNormalization
from tensorflow.keras.layers import Input,Dense,Flatten
from tensorflow.keras.models import Model
import data.actions,sim_core,deep

class FrameSim(object):
    def __init__(self,n_hidden=128):
        self.n_hidden=n_hidden
        self.n_kerns=[64,32]
        self.kern_size=[(5,5),(3,3),(3,3)]
        self.pool_size=[(4,4),(3,3),(2,2)]  

    def __call__(self,params):
        input_shape=params["input_shape"]
        img_a = Input(shape=input_shape)
        img_b = Input(shape=input_shape)
        feature_extractor = self.build_model(input_shape)
        feats_a = feature_extractor(img_a)
        feats_b = feature_extractor(img_b)
        distance =sim_core.distance_layer(feats_a,feats_b)
        model = Model(inputs=[img_a, img_b], outputs=distance)
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
    fun=center_frame
    action_dict=data.actions.get_actions(in_path,fun,out_path=None,dims=dims)
    train,test=action_dict.split()
    X,y=sim_core.pair_dataset(train)
#    raise Exception(X[0].shape)
    params={"input_shape":(*train.dim(),1)}
    make_nn=FrameSim()
    make_nn(params)

def center_frame(frames):
    center= int(len(frames)/2)
    return frames[center]

in_path="../best2/frames"
train(in_path,None)