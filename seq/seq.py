import imp
utils =imp.load_source("utils","/home/user/cf/conv_frames/utils.py")

class Dataset(object):
    def __init__(self,instances):
        self.size=len(instances)
        self.instances=instances
    
    def get_labels(self):
        return [inst.category for inst in self.instances]

    def get_instance(self,i):
        return self.instances[i]

    def get_seq(self,i):
        return self.instances[i].seq

    def get_category(self,i):
        return self.instances[i].category

    def get_person(self,i):
        return self.instances[i].person

class Instance(object):
    def __init__(self,seq,category,person):
        self.seq=seq
        self.category=category
        self.person=person 

    def sub_seq(self):
        indexes=range(len(self.seq)-1)
        sub_seqs=[self.seq[i] +self.seq[i+1] for i in indexes]
        return sub_seqs

    def __str__(self):
        cat=str(self.category)
        person=str(self.person)
        return str(self.seq) + "$" + cat +"$" + person

def create_dataset(path):
    lines=utils.read_file(path)
    instances=map(parse_instance,lines)
    return Dataset(instances)

def save_instances(out_path,dataset):
    utils.to_txt_file(out_path,dataset.instances)
    
def parse_instance(raw_instance):
    raw=raw_instance.split('$')
    seq=raw[0]
    category=int(raw[1])
    person=int(raw[2])
    return Instance(seq,category,person)

def split_dataset(path):
    dataset=create_dataset(path)
    train_path=path.replace(".seq","_train.seq")
    test_path=path.replace(".seq","_test.seq")
    insts=dataset.instances
    train_insts=[inst for inst in insts if odd_person(inst)]
    test_insts=[inst for inst in insts if not odd_person(inst)]
    save_instances(train_path,Dataset(train_insts))
    save_instances(test_path,Dataset(test_insts))

def odd_person(instance):
    return (instance.person % 2)==1

if __name__ == "__main__":
    path="/home/user/cf/seqs/dataset.seq"
    split_dataset(path)
