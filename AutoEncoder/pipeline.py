import tensorflow as tf
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pathlib

AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_data(data_dir, BATCH_SIZE):
    global CLASS_NAMES
    
    data_dir = pathlib.Path(data_dir)
    
    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])

    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))

    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    train_ds = prepare_for_training(labeled_ds, BATCH_SIZE)
    return train_ds



def get_label(file_path):
    #convert the path to a list of path components
    parts = tf.strings.split(file_path, '/')
    #The second to last is the class-directory
    return parts[-2]

def decode_img(img):
    #convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    #Use 'convert_image_dtype' to convert to floats in the [0,1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    #resize the image to the deisred size
    return tf.image.resize(img, [256,256])

def process_path(file_path):
    label = get_label(file_path)
    #load the raw data from teh file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, img#label == CLASS_NAMES

def prepare_for_training(ds, BATCH_SIZE, cache=True, shuffle_buffer_size=1000):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds