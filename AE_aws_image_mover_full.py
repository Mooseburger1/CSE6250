# How to run script
# python AE_image_mover.py file.csv bucket
# file.csv is the file to transfer images from
# bucket is the name of s3 bucket
# in this case all the files are in autoencoders folder so I have
# that hard coded below
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
    client = boto3.client(
            's3',
            aws_access_key_id="AKIAWVFQFSQH7O5PHHMJ",
            aws_secret_access_key="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"
            )
    return client

def read_csv(filepath):
    images = pd.read_csv(filepath)
    return images

def copy_images_to_s3(classification, images_with_classification):
    paths = images_with_classification['Path']
    formatted_classification = classification.replace(" ", "_").lower()
    for image_with_classification in paths:
        new_image_path = image_with_classification.replace("valid/", "").replace("train/", "").replace("/", "_")
        if "valid" in image_with_classification:
            image_type = "valid"
        else:
            image_type = "train"
        current_key = 'cse6250/' + image_with_classification
        destination_key = 'cse6250/processed/' + image_type + "/" + formatted_classification + '/' + new_image_path
        copy_image_to_s3(current_key, destination_key)
    return True

def copy_image_to_s3(current_key, dest_key):
    s3_client = configure_s3_client()
    print("copying current_key: " + current_key)
    print("to destination_key: " + dest_key)
    source_key = { 'Bucket': 'harrisjosh-project-data', 'Key': current_key}
    destination_key = { 'Bucket': 'harrisjosh-project-data', 'Key': dest_key}
    copy_cmd = s3_client.copy_object(CopySource=source_key, Bucket='harrisjosh-project-data', Key=dest_key)
    return copy_cmd

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
