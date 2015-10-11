import utils
import numpy as np

SYMBOLS="ABCDEFGHIJKLMNOPRSTUVWXYZ"

class Action(object):
    def __init__(self,category,person,frames,flat=True):
        self.category=category
        self.person=person
	self.frames=frames
        self.seq=range(len(frames))

    def set_seq(self,cls_frames):
        self.seq=[SYMBOLS[cat] for cat in cls_frames]

    def flat_frames(self):
        return [frame.reshape((1,frame.size)) for frame in self.frames]

    def __str__(self):
        cf="".join([str(cat) for cat in self.seq])
        cf+="$"+str(self.category)
        cf+="$"+str(self.person)
        return cf+"\n"

def read_action(action_path):
    action_name=get_action_name(action_path)
    category=get_category(action_name)
    person=get_person(action_name)
    all_files=utils.get_files(action_path)
    all_files=utils.append_path(action_path+"/",all_files)
    frames=utils.read_images(all_files)
    return Action(category,person,frames)

def get_action_name(action_path):
    return action_path.split("/")[-1]

def get_category(action_name):
    raw_cat=action_name.split("_")[0]
    cat=raw_cat.replace("a","")
    return int(cat)

def get_person(action_name):
    raw_person=action_name.split("_")[1]
    person=raw_person.replace("s","")
    return int(person)

if __name__ == "__main__":
   action_path="/home/user/cf/actions/a01_s01_e01_sdepth"
   action=read_action(action_path)
   print(action)
