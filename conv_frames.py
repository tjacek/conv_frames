import utils,actions
from cls.logit import LogitClassifier,FlatImages,LogitModel

def create_seq_dataset(action_dir,cls_path):
    cls=utils.read_object(cls_path)
    actions=get_actions(action_dir,cls)
    #seq_data=utils.array_to_txt(actions)
    #utils.save_string(conf.seq,seq_data)

def get_actions(action_dir,cls):
    action_files=utils.get_dirs(action_dir)
    action_files=utils.append_path(action_dir,action_files)
    return [compute_sequence(path,cls) for path in action_files]

def compute_sequence(action_path,cls):
    action=actions.read_action(action_path)
    cls_frames= [cls.test(frame) for frame in action.flat_frames()]
    action.set_seq(cls_frames)
    print(action)
    return action

if __name__ == "__main__":
    action_dir="/home/user/cf/actions/"
    cls_path="/home/user/cf/exp1/logit"
    cls=create_seq_dataset(action_dir,cls_path)
