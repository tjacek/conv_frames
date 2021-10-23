import sys
sys.path.append("..")
import data.actions

def substract(in_path,action_path,out_path):
    backgrounds=data.actions.read_actions(in_path,"color")
    action_dict=data.actions.read_actions(action_path,"color")
    clean_actions=data.actions.ActionImgs()
    for name_i,img_i in action_dict.items():
        view_j=name_i.split("_")[-1]
        background_j=backgrounds[view_j]
        print(name_i)
        diff_i=(img_i-background_j)
        diff_i[diff_i<0]=0
        clean_actions[name_i]=diff_i
    clean_actions.save(out_path)

in_path="../../Background/raw"
action_path="../../actions"
out_path="../../actions2"
substract(in_path,action_path,out_path)