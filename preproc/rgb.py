import sys
sys.path.append("..")
import re
import files

in_path="../../event1_part1/"

def is_view(path_i):
    id_i= path_i.split("/")[-1]
    id_i= re.sub(r"(\d)+","",id_i)
    print(id_i)
    return (id_i=="view")

def get_name(path_i):
    print(path_i)
    raw=path_i.split("/")[-3:]
    train =(raw[1].find("train")>=0)
    cat_i=re.sub(r"(\D)+","",raw[0].split("_")[0])
    person_i=re.sub(r"(\D)+","",raw[1])
    view_i=re.sub(r"(\D)+","",raw[2])
    name_i="%s_%d_%s_%s" % (cat_i,int(train),person_i,view_i)
    return name_i


paths=files.find_dirs(in_path,is_view)
#print(paths)
files.move_dirs(paths,"../../rgb",get_name)