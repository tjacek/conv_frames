from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.wrappers import TimeDistributed
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras import regularizers
import gen,deep

class FRAME_LSTM(object):
	def __init__(self,dropout=0.5,activ='relu',batch=False,l1=0.01):
#		if(optim_alg is None):
#			optim_alg=deep.Adam(0.00001)
		self.dropout=dropout
		self.activ=activ
		self.batch=batch
		self.l1=l1
		self.optim_alg="adam"

	def __call__(self,params):
		input_shape= (params['seq_len'],*params['dims']) 
		model=Sequential()
		n_kern,kern_size,pool_size=[64,64,64],[(5,5),(5,5),(5,5)],[(2,2),(2,2),(2,2)]
		lstm_cnn(model,n_kern,kern_size,pool_size,self.activ,input_shape)
		model.add(TimeDistributed(Flatten()))
		model.add(TimeDistributed(Dense(256)))
		if( not (self.dropout is None)):
			model.add(TimeDistributed(Dropout(self.dropout)))	
		reg=None if(self.l1  is None) else regularizers.l1(self.l1)
		model.add(TimeDistributed(Dense(128, name="first_dense",
				kernel_regularizer=reg)))

		model.add(LSTM(64, return_sequences=True, name="lstm_layer"));
		
		if(self.batch):
			model.add(GlobalAveragePooling1D(name="prebatch"))
			model.add(BatchNormalization(name="global_avg"))
		else:
			model.add(GlobalAveragePooling1D(name="global_avg"))
		model.add(Dense(params['n_cats'],activation='softmax'))

		model.compile(loss='categorical_crossentropy',
			optimizer=self.optim_alg,#keras.optimizers.Adadelta(),
			metrics=['accuracy'])
		model.summary()

def lstm_cnn(model,n_kern,kern_size,pool_size,activ,input_shape):
    for i,n_kern_i in enumerate(n_kern):
        if(i==0):
            conv_i=Conv2D(n_kern_i,kern_size[i], padding='same')
            model.add(TimeDistributed(conv_i,input_shape=input_shape))
        else:
            conv_i=Conv2D(n_kern_i,kern_size[i])
            model.add(TimeDistributed(conv_i))
        model.add(TimeDistributed(Activation(activ)))
        model.add(TimeDistributed(MaxPooling2D(pool_size=pool_size[i])))

def train(in_path):
    make_model=FRAME_LSTM()
    sampler=gen.make_lazy_sampler(in_path)
    params={'seq_len':sampler.subsample.size,
                'dims':(128,64,1),"n_cats":12}
    model=make_model(params)

in_path="../final"
train(in_path)