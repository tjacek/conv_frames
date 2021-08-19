import numpy as np
import cut,gui,files

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

def cut_frames(in_path,out_path,train_path="train_wall"):
    cut.cut_frames(in_path,train_path,out_path,cut_wall)

def make_exp(in_path,train_path="train_wall"):
    state=make_wall_state(in_path,train_path)
    gui.gui_exp(state)

def do_exp(in_path,out_path,train_path="train_wall"):
    files.make_dir(out_path)
    out_path="%s/wall" % out_path
    cut_frames(in_path,out_path,train_path)

if __name__ == "__main__":
    in_path="actions"
#    make_exp(in_path,train_path="train_wall")
    in_path="../../3DHOI2/raw/"
    out_path="../../3DHOI4/B"
    train_path="train_wall"
    do_exp(in_path,out_path,train_path)