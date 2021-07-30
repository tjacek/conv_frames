import sys
sys.path.append("..")
from PyQt5 import QtGui, QtCore, QtWidgets
import cv2
import gui, data.actions

class ActionState(object):
    def __init__(self, actions_dict):
        self.actions_dict=actions_dict
        self.path="path"

    def __getitem__(self,frame_i):
        print(frame_i)

    def show(self,name_i,text_i):
        img_i=self.actions_dict[name_i]
        cv2.imshow(name_i,img_i)

    def keys(self):
        return self.actions_dict.keys()

    def save(self,path_i):
        print(path_i)

def make_action_state(in_path):
    actions_dict=data.actions.read_actions(in_path)
    return ActionState(actions_dict)

if __name__ == "__main__":
    in_path="../../../Downloads/AA/depth/actions"
    state=make_action_state(in_path)
    gui.gui_exp(state)