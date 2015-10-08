import imp
utils =imp.load_source("utils","/home/user/cf/conv_frames/utils.py")

class Instances(object):
    def __init__(self,seqs,cats,persons):
        self.size=len(seqs)
        self.seqs=seqs
        self.cats=cats
        self.persons=persons

    def get_instance(self,i):
        return self.seqs[i],self.cats[i],self.persons[i]

    def to_tuple(self):
        return [self.get_instance(i) for i in range(self.size)]

def parse_instances(path):
    lines=utils.read_file(path)
    instances=map(parse_instance,lines)
    seqs=[inst[0] for inst in instances]
    cats=[inst[1] for inst in instances]
    persons=[inst[2] for inst in instances]
    return Instances(seqs,cats,persons)

def save_instances(out_path,instances):
    lines=instances.to_tuple()
    txt=utils.array_to_txt(lines)
    utils.save_string(out_path,txt)

def parse_instance(raw_instance):
    raw=raw_instance.split('$')
    seq=raw[0]
    category=int(raw[1])
    person=int(raw[2])
    return seq,category,person

def seq_to_string(seq):
    return seq[0] + "$" + seq[1] +"$" +seq[2]


if __name__ == "__main__":
    path="/home/user/cf/seqs/dataset.seq"
    inst=parse_instances(path)
    save_instances(path.replace(".seq","_.seq"),inst)
