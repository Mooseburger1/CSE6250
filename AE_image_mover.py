# How to run script
# python AE_image_mover.py file.csv 
# file.csv is the file to transfer images from
import pandas as pd
import numpy as np
from shutil import copyfile
import os

csv_src = sys.argv[0]
img_src = sys.argv[1]

new_folder = sys.argv[0].split(".")[0]
os.rmdir(new_folder)
os.mkdir(new_folder)

#df = pd.read_csv(Pleural_Other.csv")
df = pd.read_csv("autoencoders/"+str(csv_src))
count_f = 0
count_l = 0
for path in df['Path']:
    img_name = path.split("/")[-1]
    if "frontal" in img_name:
        name = "frontal_"+count_f+".jpg"
        count_f +=1
    else:
        name = "lateral_"+count_l+".jpg"
        count_l +=1
    copyfile(path,new_folder+"/"+name)
    
    #print(path)

