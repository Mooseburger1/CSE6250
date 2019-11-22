# How to run script
#python3 AE_aws_image_mover_full.py valid-full-test.csv
import pandas as pd
import numpy as np
import os
import sys
import boto3

#valid-full-test.csv
#train-full-test.csv

def classification_list():
    return ['No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices']

def configure_s3_client():
    aws_access_key_id = "AKIAWVFQFSQH7O5PHHMJ"
    aws_secret_key_id = "tJ4DEIYjJ3a86gdzMBcVc+Qn2DgSoRflx7WKH+Z6"
    return boto3.resource('s3')

def read_csv(filepath):
    images = pd.read_csv(filepath)
    return images

def copy_images_to_s3(classification, images_with_classification):
    paths = images_with_classification['Path']
    for image_with_classification in paths:
        current_key = 'cse6250/' + image_with_classification
        destination_key = 'cse6250/processed/' + classification + '/' + image_with_classification
        copy_image_to_s3(current_key, destination_key)
    return True

def copy_image_to_s3(current_key, dest_key):
    s3_client = configure_s3_client()
    print("copying current_key: " + current_key)
    print("to destination_key: " + dest_key)
    #source_key = { 'Bucket': 'harrisjosh-project-data', key: current_key}
    #destination_key = { 'Bucket': 'harrisjosh-project-data', key: dest_key}
    #s3.meta.client.copy_object(CopySource=source_key, Bucket='harrisjosh-project-data', Key=dest_key)
    return True

def main():
    filepath = sys.argv[1]
    print(("reading from file: " + str(filepath)))
    images = read_csv(filepath)
    #iterate through classifications
    for classification in classification_list():
        #pull images with 1.0 score for classification
        relevant_images = images[classification] == 1.0
        images_with_classification = images[relevant_images]
        print("processing: " + str(len(images_with_classification)) + " images with classification " + classification)

        #copy images for classification to processed folder
        copy_images_to_s3(classification, images_with_classification)
    return True

if __name__ == "__main__":
    main()
