import sys
sys.path.append("..")
import numpy as np
import cv2
import data.imgs
 
def kmeans(in_path,out_path,k=5):
    read=data.imgs.ReadFrames(color="color")
    def fun(name_i,img_i):
        print(name_i)
        pixel_values = img_i.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        labels = labels.flatten()
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(img_i.shape)
        return segmented_image
    data.imgs.transform_lazy(in_path,out_path,fun,read,
            recreate=True,single=True)

in_path="../../cc2/frames"
out_path="segm"
kmeans(in_path,out_path)