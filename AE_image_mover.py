# How to run script
# python AE_image_mover.py file.csv 
# file.csv is the file to transfer images from
# in this case all the files are in autoencoders folder so I have
# that hard coded below
import pandas as pd
import numpy as np
from shutil import copyfile
import os
import sys
import boto3


aws_access_key_id = "AKIAWVFQFSQH7O5PHHMJ"
aws_secret_key_id = "tJ4DEIYjJ3a86gdzMBcVc+Qn2DgSoRflx7WKH+Z6"
s3 = boto3.resource('s3')



csv_src = sys.argv[1]

new_folder = sys.argv[1].split(".")[0]
try:
    os.rmdir(new_folder)
except:
    pass
try:
    os.mkdir(new_folder)
except:
    pass

#df = pd.read_csv("Pleural_Other.csv")
df = pd.read_csv("autoencoders/"+str(csv_src)) # hard coded to be in this folder, can change this if we want
count_f = 0
count_l = 0
for path in df['Path']:
    img_name = path.split("/")[-1]
    if "frontal" in img_name:
        name = "frontal_"+str(count_f)+".jpg"
        count_f +=1
    else:
        name = "lateral_"+str(count_l)+".jpg"
        count_l +=1
    copyfile(path,new_folder+"/"+name) # python way to copy new file 
    
    #print(path)

