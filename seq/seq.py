import imp
utils =imp.load_source("utils","/home/user/cf/conv_frames/utils.py")

class Instances(object):
    def __init__(self,seqs,cats,persons):
        self.size=len(seqs)
        self.seqs=seqs
        self.cats=cats
        self.persons=persons

def parse_dataset(path):
    lines=utils.read_file(path)
    instances=map(parse_instance,lines)
    seqs=[inst[0] for inst in instances]
    cats=[inst[1] for inst in instances]
    persons=[inst[2] for inst in instances]
    return Instances(seqs,cats,persons)

def parse_instance(raw_instance):
    raw=raw_instance.split('$')
    seq=raw[0]#get_frames(raw[0])
    category=int(raw[1])
    person=int(raw[2])
    return seq,category,person

if __name__ == "__main__":
    path="/home/user/cf/seqs/dataset.seq"
    print(parse_dataset(path))
