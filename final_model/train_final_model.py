import sys
import os
sys.path.append('..')

from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
import numpy as np
from AutoEncoder.CheXpert import Model1, Model2
from AutoEncoder.pipeline import make_dataset2
import matplotlib.pyplot as plt
import shutil
import time
import datetime
import argparse
import glob
import io


class_names={ 0:'Lung_Lesion',
              1:'Atelectasis',
              2: 'No_Finding',
              3: 'Edema' ,
              4:'Lung_Opacity' ,
              5: 'Cardiomegaly',
              6: 'Pleural_Other',
              7 :'Fracture',
              8: 'Pneumonia',
              9: 'Enlarged_Cardiomediastinum',
              10: 'Pleural_Effusion' ,
              11: 'Pneumothorax',
              12 :'Support_Devices',
              13 :'Consolidation'}

def plot_image(predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label, img
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  

  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label
  plt.grid(False)
  plt.xticks(range(14))
  plt.yticks([])
  thisplot = plt.bar(range(14), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

def plot_it(prediction, true, img):
    fig = plt.figure(figsize=(18,9))
    plt.subplot(1,2,1)
    plot_image(prediction, true, img)
    plt.subplot(1,2,2)
    plot_value_array(prediction,  true)
    return fig

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image
    
    

def cheatsheet():
    inputs = tf.keras.layers.Input(shape=[299,299,3])
    
    reconstruction_differences = []
    
    for model in AE_models:
        x = model(inputs)
        diff = tf.math.subtract(inputs, x)
        reconstruction_differences.append(diff)
        
    cheat_sheet = tf.keras.layers.Concatenate()(reconstruction_differences)
    return tf.keras.Model(inputs=inputs, outputs=cheat_sheet,name='cheatsheet')

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

#training function
@tf.function
def train_step(x_train, y_train):
    with tf.GradientTape() as tape:
        #forward prop
        predictions = model(x_train, training=True)
        #calculate loss
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_train+0.0, predictions, from_logits=False)
        #backwards prop - calculate gradients
        grads = tape.gradient(loss, model.trainable_variables)
        #update weights
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #update loss metric with current batch loss
    train_loss_metric(loss)
    train_acc(y_train, predictions)

@tf.function
def valid_step(x_val, y_val):
    predictions = model(x_val, training=True)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_val+0.0, predictions, from_logits=False)

    valid_loss_metric(loss)
    valid_acc(y_val, predictions)

#fit function       
def fit(model, optimizer, epochs, train, test):

    print('\n\nTraining Starting @ {}'.format(datetime.datetime.now()))
    
    #Train for specified number of epochs
    for epoch in range(int(epochs)+1):
        print('EPOCH: {}'.format(epoch))
        #forward prop and backwards prop for current epoch on training batches

        for (x_train, y_train, _) in train:
            # if epoch == 0:
            #     tf.summary.trace_on(graph=True, profiler=False)
            train_step(x_train, y_train)

        #save model checkpoint every 10 epochs and write Tensorboard summary updates
        if (epoch) % 1 == 0:
            for (x_val, y_val, _) in test:
                valid_step(x_val, y_val)
                
            #checkpoint model
            checkpoint.save(file_prefix=checkpoint_prefix)
            
            #predict on test image
            pred_y = model(test_image)

            figure = plot_it(pred_y, test_label, test_image)
            
            #write loss, test image, and predicted image to Tensorboard logs
            with train_summary_writer.as_default():
                # if epoch==0:
                #     tf.summary.trace_export(name="my_func_trace", step=0)
                tf.summary.scalar('train_loss', train_loss_metric.result(), step=epoch)
                tf.summary.scalar('train_accuracy', train_acc.result(), step=epoch)
                tf.summary.image('test_image', plot_to_image(figure), step=epoch)
                


            with valid_summary_writer.as_default():
                tf.summary.scalar('valid_loss', valid_loss_metric.result(), step=epoch)
                tf.summary.scalar('valid_accuracy', valid_acc.result(), step=epoch)

            #Log training loss to console for monitoring as well
            print('Epoch [%s]: mean loss (train/val): [%s]/[%s]' % (epoch, train_loss_metric.result().numpy(),valid_loss_metric.result().numpy()))
        else:
            print('Epoch [%s]: mean loss (train only): [%s]' % (epoch, train_loss_metric.result().numpy()))
        #reset the loss metric after each epoch
        train_loss_metric.reset_states()
        valid_loss_metric.reset_states()
        train_acc.reset_states()
        valid_acc.reset_states()
    model.save('model.h5')

#################### MAIN SCRIPT STARTS HERE #######################

'''Section for CLI arguments and descriptions'''
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--models_directory', dest='models', help="Directory location of trained AutoEncoder Models")
parser.add_argument('-t', '--tfrecords', dest='tfrs', help='Directory location of tfrecords')
parser.add_argument('-o', '--output_directory', dest='output', help="Directory to save all data")
parser.add_argument('-e', '--epochs', dest='epochs', help='Number of training epochs', default=100)
parser.add_argument('-b', '--batch', dest='batches', help='Batch Size for training data', default=64)
parser.add_argument('-n', '--nuumber_model_to_train', dest='model_number', help='Which architecture to train [1,2]')

#parse CLI arguments
args = parser.parse_args()



'''Data Input/Pipeline and Model Section'''
#input pipeline
train_path = os.path.join(args.tfrs, "train")
train_dataset = make_dataset2(train_path)

valid_path = os.path.join(args.tfrs, "valid")
valid_dataset = make_dataset2(valid_path)


'''Test Images to log on Tensorboard'''
test_img = valid_dataset.take(1)

for x,y,z in test_img:
    test_image = x[0]
    test_label = y[0]
    _ = z[0]





'''Restore Inception_ResNet & Trained AutoEncoders and Create Cheatsheet Generator'''
#Inception ResNet
print('Restoring InceptionResNetV2 Model')
inception_res = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(299,299,3))
inception_res.trainable = False

#List of Trained AutoEncoder Models
print('Restoring Trained AutoEncoder Models')
list_of_model_paths = glob.glob(args.models + '/*/model/*.h5')

AE_models = restore_models(list_of_model_paths)



'''Model and Optimizer'''
#Model Architecture
if int(args.model_number) == 1:
    model = Model1(inception_res, cheatsheet)
elif int(args.model_number) == 2:
    model = Model2()
else:
    print('Model architecture parameter must be 1 or 2 - Program terminating')
    sys.exit()

#Optimizer
optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

'''Metrics'''
#Declare loss metrics
train_loss_metric = tf.keras.metrics.Mean('train_loss')
train_acc = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
valid_loss_metric = tf.keras.metrics.Mean('valid_loss')
valid_acc = tf.keras.metrics.SparseCategoricalAccuracy('valid_accuracy')


'''Admin Section'''
#Tensorboard logging
train_current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_path = os.path.join(args.output, 'logs')
train_log_dir = os.path.join(train_log_path, train_current_time)
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

valid_current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
valid_log_path = os.path.join(args.output, 'logs')
valid_log_dir = os.path.join(valid_log_path, valid_current_time)
valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

#Model Checkpoint Object
checkpoint_prefix = os.path.join(args.output, "checkpoints/ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
restore_dir = os.path.join(args.output, 'checkpoints')

print("Restoring latest checkpoint from {} if available".format(restore_dir))
checkpoint.restore(tf.train.latest_checkpoint(restore_dir))


'''Model Training Section'''
#Train model
fit(model, optimizer, args.epochs, train_dataset, valid_dataset)