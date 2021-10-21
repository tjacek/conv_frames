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

paths=files.find_dirs(in_path,is_view)
print(paths)
