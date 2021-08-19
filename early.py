import numpy as np
import data.imgs,tc_nn,files,learn

class EarlyReader(object):
    def __init__(self,read,ratio=0.1,n_split=1):
        self.ratio=ratio
        self.read=read
#        self.dim=dim
        self.n_split=n_split

    def __call__(self,in_path):
#        frame_seqs=data.imgs.read_frame_seqs(in_path,n_split=self.n_split)
        frame_seqs=self.read(in_path)
        mean_size= np.median(frame_seqs.seqs_len())
        seq_len=self.get_fraction(mean_size) 
        sample=data.imgs.MinLength(seq_len)
        def helper(seq_i):
            if(self.ratio<0.99):
                start_i= self.get_fraction(len(seq_i))
                start_seq_i=seq_i[:start_i]
            else:
            	start_seq_i=seq_i
            final_seq_i=sample(start_seq_i)
            return final_seq_i
        frame_seqs.transform(helper,new=False,single=False)
#        frame_seqs.scale(self.dim)
        return frame_seqs

    def get_fraction(self,len_i):
        return int(np.ceil(len_i*self.ratio))

def single_exp(frame_path,out_path,n_epochs=100,ratio=0.1):
    read=tc_nn.SimpleRead(dim=(64,128),preproc=data.imgs.Downsample())
    make_nn=tc_nn.TC_NN(n_hidden=200,batch=True)#,loss='mean_squared_error')
    read=EarlyReader(read,ratio=ratio)#,dim=(64,64))
    train=learn.Train(tc_nn.to_dataset,make_nn,read=read,batch_size=8)
    extract=learn.Extract(make_nn,read)
    files.make_dir(out_path)
    nn_path,feath_path="%s/nn" % out_path,"%s/feats" % out_path
    train(frame_path,nn_path,n_epochs=n_epochs)
    extract(frame_path,nn_path,feath_path)

def ens_exp(frame_path,out_path,n_epochs=100):
    files.make_dir(out_path)
    for i in range(10):
        ratio_i=(i+1)*0.1
        out_i="%s/%d" % (out_path,(i+1))
        single_exp(frame_path,out_i,n_epochs,ratio_i)

in_path="../3DHOI4/B/frames"
out_path="../3DHOI4/B/early"

ens_exp(in_path,out_path,n_epochs=50)