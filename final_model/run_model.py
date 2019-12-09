import pandas as pd
import numpy as np
import tensorflow as tf
import os
from pipeline import make_dataset
import glob
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.models import load_model
import argparse
from CheXpert import Model2, cheatsheet
import time
import matplotlib.pyplot as plt


def rename_layers(model, ext):
    for layer in model.layers:
        layer._name = layer.name + ext

def restore_models(paths):
    restored_models = []
    for pos, path in enumerate(paths):
        model = load_model(path)
        model._name = 'AE_{}'.format(pos)
        rename_layers(model, str(pos))
        
        model.trainable=False
        
        restored_models.append(model)
        
    return restored_models

def final_model():
    inputs=tf.keras.layers.Input(shape=[299,299,3])

    
    x_trans = inception_res(inputs)
    x_trans = tf.keras.layers.Flatten()(x_trans)
    x_cheat = cheatsheet_generator(inputs)
    x_cheat = tf.keras.layers.Flatten()(x_cheat)
    
    x = tf.keras.layers.Concatenate()([x_trans, x_cheat])
    
    x = fully_connected(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x, name = 'Code of Conduct Model')



'''Section for CLI arguments and descriptions'''
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_directory', dest='input_', help="Directory location of tfrecords files of images")
parser.add_argument('-m', '--models_directory', dest='models', help="Parent Directory of all trained autoencoders")
parser.add_argument('-o', '--output_directory', dest='output', help="Directory to save all data")
parser.add_argument('-b', '--batch', dest='batches', help='Batch Size for training data', default=64)

#parse CLI arguments
args = parser.parse_args()

print('Loading data from TFRecords files')
'''load data from tfrecords'''
tfrecords_dir = os.path.join(args.input_, '*')
data = make_dataset(tfrecords_dir, int(args.batches))
print('Testing deserialization - Rendering one sample image')
print('Close image to continue')
time.sleep(3)

sample = data.take(1)
for x,y,z in sample:
    img = x[0]
    label = y[0]
    cls_ = z[0]

plt.figure()
plt.imshow(img)
plt.xlabel('Example Image from TFRecords data')
plt.title('Classification: {}'.format(cls_))
plt.show()



models_dir = os.path.join(args.models, '*/*.h5')
list_of_model_paths = glob.glob(models_dir)
list_of_aes = [x for x in list_of_model_paths if 'fc.h5' not in x]
print('AutoEncoder saved models found: ', list_of_aes)
path_to_fully_connected = [x for x in list_of_model_paths if 'fc.h5' in x]
print('Full Connected model found: ', path_to_fully_connected)

print('Restoring AutoEncoder Models')
'''restore all 14 autoencoder models and instantiate the cheat sheet generator model'''
AE_models = restore_models(list_of_aes)
cheatsheet_generator = cheatsheet(AE_models)

print('Restoring InceptionResNetV2')
'''restore the InceptionResNetV2 model'''
inception_res = InceptionResNetV2(include_top=False,
                                  weights='imagenet',
                                  input_shape=(299,299,3))

inception_res.trainable = False

print('Restoring Fully Connected Layer')
'''restore the fully connected trained model'''
fully_connected = load_model(path_to_fully_connected[0])
fully_connected.trainable = False



CoC = final_model()
print('\n\n\n')
print(CoC.summary())
print('\n\n')


time.sleep(3)
print('Predicting................')
print('Depending on your computer - this might take a minute or two')
print("Theres 442 million parameters afterall")
print("If you have Tensorflow-GPU, you'll be A-Ok")


preds = []
labels = []
classes = []

acc_metric = tf.keras.metrics.CategoricalAccuracy('train_accuracy')

for img, lab, cls_ in data:
    pred = CoC(img)
    
    preds.extend(pred.numpy())
    labels.extend(lab.numpy())
    classes.extend(cls_.numpy())
    
    acc_metric(lab, pred)



print('Accuracy is: ', acc_metric.result().numpy())

data_dict = {'class':classes, 'predictions (softmax output)':preds, 'True Label (one-hot-encoded)':labels}

df = pd.DataFrame(data_dict)
output_dir = os.path.join(args.output, 'Predictions.csv')
df.to_csv(output_dir, index=False)
print('Results saved locally to "Predictions.csv"')