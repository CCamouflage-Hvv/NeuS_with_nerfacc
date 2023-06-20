import cv2
from glob import glob
import os
# a = cv2.imread("D:/NeRF_/repos/NeuS/my_data/DJIWC_NeuS_repos_5_views\image\DJI_20210425153029_0057.jpg")
# print(a.shape)

data_dir = "D:/NeRF_/repos/NeuS/my_data/DJIWC_NeuS_repos"
sorted_image_lis = sorted(glob(os.path.join(data_dir, 'image/*.jpg')))
image_lis = (glob(os.path.join(data_dir, 'image/*.jpg')))
for item in sorted_image_lis:
    print(item)
#print(sorted_image_lis)