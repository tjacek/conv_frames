import sys
sys.path.append("..")
from PyQt5 import QtGui, QtCore, QtWidgets
import json,os.path
import cv2,numpy as np
from ast import literal_eval
import gui, data.actions

class ActionState(object):
    def __init__(self, actions_dict,train_data,train_path):
        self.actions_dict=actions_dict
        self.path=train_path
        self.train_data=train_data

    def __getitem__(self,name_i):
        return str(self.train_data[name_i])

    def show(self,name_i,text_i):
        img_i=self.actions_dict[name_i]
        position=literal_eval(text_i)
        self.train_data[name_i]=position
        new_img_i=cut_rect(img_i,position)
        cv2.imshow(name_i,new_img_i)

    def keys(self):
        return self.actions_dict.keys()

    def save(self,path_i):
        with open(path_i, 'w') as f:
            json.dump(self.train_data, f)

def cut_rect(img_i,position):
	if(np.product(position)==0):
		return img_i
	position=np.array(position)
	if(type(position)==np.ndarray):
		position=position.astype(int)
	position[position<0]=0
	x0,y0=position[0],position[1]
	x1,y1=x0+position[2],y0+position[3]
	img_i=img_i.copy()
	img_i[:x1,y0:y1]=0
	return img_i

def make_train(actions_dict):
    return {name_i:[0,0,0,0] for name_i in actions_dict.keys()}	

def make_action_state(in_path,train_path="train"):
    actions_dict=data.actions.read_actions(in_path)
    if(os.path.isfile(train_path)):
        train_data=json.load(open(train_path))
    else:
        train_data=make_train(actions_dict)
    return ActionState(actions_dict,train_data,train_path)

if __name__ == "__main__":
    in_path="../../../Downloads/AA/depth/actions"
    state=make_action_state(in_path)
    gui.gui_exp(state)