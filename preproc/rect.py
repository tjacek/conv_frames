import numpy as np
import cut,gui,files

class ReactState(cut.ActionState):
	def __init__(self, actions_dict,train_data,train_path,cut):
		super(ReactState, self).__init__( actions_dict,train_data,train_path,cut_rect)
		
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
	img_i[x0:x1,y0:y1]=200
	return img_i

def true_cut(img_i,position_i):
    x0,y0=position_i[0],position_i[1]
    x1,y1=x0+position_i[2],y0+position_i[3]     
    return img_i[x0:x1,y0:y1]

def make_rect_state(in_path,train_path="train_rect"):
    return cut.make_action_state(in_path,train_path,cut_fun=cut_rect)

def cut_frames(in_path,out_path,train_path="train_rect"):
    cut.cut_frames(in_path,train_path,out_path,true_cut)

if __name__ == "__main__":
    in_path="actions"
#    state=make_rect_state(in_path,train_path="train_rect")
#    gui.gui_exp(state)
    in_path="../../3DHOI2/wall"
    out_path="../../3DHOI2/try3"
    files.make_dir(out_path)
    train_path="train_rect"
    out_path="%s/frames" % out_path
    cut_frames(in_path,out_path,train_path)