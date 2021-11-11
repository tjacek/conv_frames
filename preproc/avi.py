import sys
sys.path.append("..")
import cv2
import files,data.imgs

def to_rgb(in_path,out_path):
    files.make_dir(out_path)	
    for path_i in files.top_files(in_path):
        postfix=path_i.split(".")[-1]
        if(postfix=="avi"):
            name_i= files.get_name(path_i)
            gest,person,action,cat=name_i.split("_")
            name_i="_".join([cat,person,action,gest])
            print(name_i)	
            vidcap = cv2.VideoCapture(path_i)
            success,frames=True,[]
            while(success):
                success,image = vidcap.read()
                if(success):
                    frames.append(image)                
            out_i="%s/%s" % (out_path, name_i)
            data.imgs.save_frames(out_i,frames)

in_path="../../raw"
out_path="../../rgb"
to_rgb(in_path,out_path)