import scipy.io
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import numpy as np
from numpy import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import cv2

data_dict = {'Lung_Lesion': 0,
 'Atelectasis': 1,
 'No_Finding': 2,
 'Edema': 3,
 'Lung_Opacity': 4,
 'Cardiomegaly': 5,
 'Pleural_Other': 6,
 'Fracture': 7,
 'Pneumonia': 8,
 'Enlarged_Cardiomediastinum': 9,
 'Pleural_Effusion': 10,
 'Pneumothorax': 11,
 'Support_Devices': 12,
 'Consolidation': 13}


def equalize(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(img)
    return cl

def _load_image(path):
    image = cv2.imread(path,3)
    if image is not None:
        image = equalize(image)
        image = cv2.resize(image, (299,299))
        return image
    return None

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _build_examples_list(input_folder, seed):
    examples = []
    for classname in os.listdir(input_folder):
        class_dir = os.path.join(input_folder, classname)
        if (os.path.isdir(class_dir)):
            for filename in os.listdir(class_dir):
                filepath = os.path.join(class_dir, filename)
                example = {
                    'classname': classname,
                    'path': filepath,
                    'label': data_dict[classname]
                }
                examples.append(example)

    random.seed(seed)
    random.shuffle(examples)
    return examples


def _split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]


def _get_examples_share(examples, training_split):
    examples_size = len(examples)
    len_training_examples = int(examples_size * training_split)

    return np.split(examples, [len_training_examples])


def _write_tfrecord(examples, output_filename):
    writer = tf.io.TFRecordWriter(output_filename)
    for example in tqdm(examples):
        try:
            image = _load_image(example['path'])
            if image is not None:
                encoded_image_string = cv2.imencode('.jpg', image)[1].tostring()
                feature = {
                    'label': _bytes_feature(tf.compat.as_bytes(str(example['label']))),
                    'image': _bytes_feature(tf.compat.as_bytes(encoded_image_string)),
                    'classname': _bytes_feature(tf.compat.as_bytes(example['classname']))
                }

                tf_example = tf.train.Example(features = tf.train.Features(feature=feature))
                writer.write(tf_example.SerializeToString())
        except Exception as inst:
            print(inst)
            pass
    writer.close()

def _write_sharded_tfrecord(examples, number_of_shards, base_output_filename, is_training = True):
    sharded_examples = _split_list(examples, number_of_shards)
    for count, shard in tqdm(enumerate(sharded_examples, start = 1)):
        output_filename = '{0}_{1}_{2:02d}of{3:02d}.tfrecord'.format(
            base_output_filename,
            'train' if is_training else 'test',
            count,
            number_of_shards 
        )
        _write_tfrecord(shard, output_filename)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', dest='input_', help="Directory of training images")
    parser.add_argument('-o', '--output_directory', dest='output', help='Directory to save TFRecords')
    parser.add_argument('-s', '--shards', dest='shards', help="Number of TFRecord shards to output", default=2)
 

    args = parser.parse_args()



    #list of dictionaries {classname:'' , path:''}
    train_list = _build_examples_list(args.input_, 123)

    training_examples, _ = _get_examples_share(train_list,1.0)


    print("Creating training shards", flush = True)
    _write_sharded_tfrecord(training_examples, number_of_shards=int(args.shards), base_output_filename=args.output)
