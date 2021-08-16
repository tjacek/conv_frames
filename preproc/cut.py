import sys
sys.path.append("..")
from PyQt5 import QtGui, QtCore, QtWidgets
import json,os.path
import cv2,numpy as np
from ast import literal_eval
import gui, data.actions,data.imgs

class ActionState(object):
    def __init__(self, actions_dict,train_data,train_path,cut_fun):
        self.actions_dict=actions_dict
        self.path=train_path
        self.train_data=train_data
        self.cut_fun=cut_fun

    def __getitem__(self,name_i):
        return str(self.train_data[name_i])

    def show(self,name_i,text_i):
        img_i=self.actions_dict[name_i]
        position=literal_eval(text_i)
        self.train_data[name_i]=position
        new_img_i=self.cut_fun(img_i,position)
        cv2.imshow(name_i,new_img_i)

    def keys(self):
        return self.actions_dict.keys()

    def save(self,path_i):
        with open(path_i, 'w') as f:
            json.dump(self.train_data, f)

class TrainDec(object):
    def __init__(self,train_data,cut_fun):
        self.data=train_data
        self.cut_fun=cut_fun

    def __contains__(self, name_i):
        return (np.product(self.data[name_i])!=0)

    def __call__(self,name_i,img_i):
        position_i=self.data[name_i]
        return self.cut_fun(img_i,position_i)

def cut_actions(in_path,train_path,out_path,cut_fun):
    actions_dict=data.actions.read_actions(in_path)
    train_data=TrainDec(read_train(train_path),cut_fun)
    new_actions=data.actions.ActionImgs()
    for name_i,action_i in actions_dict.items():
        if(name_i in train_data):
            new_actions[name_i]= train_data(name_i,action_i)
    new_actions.save(out_path)

def cut_frames(in_path,train_path,out_path,cut_fun):
    frame_seqs=data.imgs.read_frame_seqs(in_path,n_split=1)
    train_data=TrainDec(read_train(train_path),cut_fun)
    new_frames=data.imgs.FrameSeqs()
    for name_i,seq_i in frame_seqs.items():
        if(name_i in train_data):
            seq_i=[ train_data(name_i,frame_i)  for frame_i in seq_i]
        new_frames[name_i]=seq_i
    new_frames.save(out_path) 

def make_train(actions_dict,default_value):
    return {name_i:default_value for name_i in actions_dict.keys()}	

def read_train(train_path):
    return json.load(open(train_path))

def make_action_state(in_path,train_path="train",cut_fun=None,default_value=None):
    actions_dict=data.actions.read_actions(in_path)
    if(os.path.isfile(train_path)):
        train_data=read_train(train_path)#json.load(open(train_path))
    else:
        print("make new train dataset %s" % train_path)
        train_data=make_train(actions_dict,default_value)
    return ActionState(actions_dict,train_data,train_path,cut_fun)

if __name__ == "__main__":
    in_path="../../../Downloads/AA/depth/actions"
    state=make_action_state(in_path)
    gui.gui_exp(state)
#    cut_actions(in_path,"train","actions")
#    in_path="../../../Downloads/AA/depth/rename"
#    cut_frames(in_path,"train","frames")