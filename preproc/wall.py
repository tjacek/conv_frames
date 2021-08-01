import numpy as np
import cut,gui

class WallState(cut.ActionState):
	def __init__(self, actions_dict,train_data,train_path,cut):
		super(ReactState, self).__init__( actions_dict,train_data,train_path,cut_wall)
		
def cut_wall(img_i,position):
	if(np.product(position)==0):
		return img_i
	print(np.amax(img_i))
	position=position[0]
	print(position)
	img_i=img_i.copy()
	img_i[img_i>position]=0
	return img_i

def make_wall_state(in_path,train_path="train_wall"):
    return cut.make_action_state(in_path,train_path,cut_fun=cut_wall,default_value= [0])

if __name__ == "__main__":
    in_path="actions"
    state=make_wall_state(in_path)
    gui.gui_exp(state)