# How to run script
# python AE_image_mover.py file.csv bucket
# file.csv is the file to transfer images from
# bucket is the name of s3 bucket
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

copy_source = {
    'Bucket': 'Bucket',
    'Key': 'folder/training/images.csv'
}

csv_src = sys.argv[1]
bucket = sys.argv[2]

new_folder = sys.argv[1].split(".")[0]


#df = pd.read_csv("Pleural_Other.csv")
df = pd.read_csv("autoencoders/"+str(csv_src)) # hard coded to be in this folder, can change this if we want
count_f = 0
count_l = 0
for path in df['Path']:
    rand = np.random.randint(1,10)
    if rand < 4:
        pre = 'valid/'
    else:
        pre = 'train/'
    img_name = path.split("/")[-1]
    if "frontal" in img_name:
        name = "/frontal_"+str(count_f)+".jpg"
        count_f +=1
    else:
        name = "/lateral_"+str(count_l)+".jpg"
        count_l +=1

    copy_source = {
        'Bucket': bucket,
        'Key': path  # might need to modify this depending on how we setup the s3 storage
    }
    s3.meta.client.copy(copy_source, bucket, pre + new_folder+name)
    #copyfile(path,new_folder+"/"+name) # python way to copy new file 
    
    #print(path)
    # secret key
    #eIDnM0x1FbhHQxCLlGEhvftkq5WM2AqklPtJEPie