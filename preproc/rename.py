import sys
sys.path.append("..")
import shutil
import os,os.path,re
import files

def rename_action(in_path,out_path):
    paths=files.top_files(in_path)
    names=[ os.path.basename(path_i)for path_i in paths]
    persons=get_person_map(names)
    def helper(name_i):
        raw=name_i.split("_")
        person_i=persons[raw[-2].lower()]
        cat_i=re.sub('[^0-9]','',raw[-1])
        print(raw[-1])
        print(cat_i)
        return "%s_%s_1"  % (cat_i,person_i)	
    name_map={ name_i:helper(name_i) for name_i in names}
    print(len(name_map))
    files.make_dir(out_path)
    for old_i,new_i in name_map.items():
        old_i="%s/%s" % (in_path,old_i)
        new_i="%s/%s" % (out_path,new_i) 	
        files.make_dir(new_i)
        for in_j in files.top_files(old_i):
            out_j= "%s/%s" % ( new_i,in_j.split("/")[-1] )	
            shutil.move(in_j, out_j)
#        print(old_i)
#        print(new_i)	

def get_person_map(names):
    persons=[name_i.split("_")[-2].lower()  for name_i in names]
    persons=list(set(persons))
    persons.sort()
    size=len(persons)/2
    def helper(i):
        return 2*(i-size) if(i>size) else (2*i+1)
    persons={ person_i: int(helper(i)) 
                for i,person_i in enumerate(persons)}
    return persons

in_path="../../../Downloads/AA/depth/depth_sampled"
out_path="../../../Downloads/AA/depth/rename"
rename_action(in_path,out_path)