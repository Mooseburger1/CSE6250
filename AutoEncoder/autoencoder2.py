import tensorflow as tf
import numpy as np
from CheXpert import XrayAE_Functional
from pipeline import make_dataset
import matplotlib.pyplot as plt
import shutil
import time
import datetime
import argparse
import os
import sys


#training function
@tf.function
def train_step(x_train, y_train):
    with tf.GradientTape() as tape:
        #forward prop
        reconstruction = model(x_train, training=True)
        #calculate loss
        loss = mse_loss_fn(y_train, reconstruction)
        #backwards prop - calculate gradients
        grads = tape.gradient(loss, model.trainable_variables)
        #update weights
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #update loss metric with current batch loss
    loss_metric(loss)
        

#fit function       
def fit(model, optimizer, epochs, train, test):

    print('\n\nTraining Starting @ {}'.format(datetime.datetime.now()))
    
    #Train for specified number of epochs
    tf.summary.trace_on(graph=True, profiler=False)
    for epoch in range(epochs):
        
        #forward prop and backwards prop for current epoch on training batches
        for (x_train, y_train, _) in train:
            train_step(x_train, x_train)
        #save model checkpoint every 10 epochs and write Tensorboard summary updates
        #if (epoch + 1) % 10 == 0:
            #checkpoint model
        checkpoint.save(file_prefix=checkpoint_prefix)
            #predict on test image
        pred_y = model(test_img)
            #write loss, test image, and predicted image to Tensorboard logs
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss_metric.result(), step=epoch)
            tf.summary.image('original', test_img, max_outputs=10, step=epoch)
            tf.summary.image('predicted', pred_y, max_outputs=10, step=epoch)
            tf.summary.trace_export(name="train_step", step=0)
            #Log training loss to console for monitoring as well
        print('Epoch [%s]: mean loss [%s]' % (epoch, loss_metric.result().numpy()))
        #reset the loss metric after each epoch
        loss_metric.reset_states()




'''Section for CLI arguments and descriptions'''
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_directory', dest='input_', help="Directory location of training images", default='.')
parser.add_argument('-o', '--output_directory', dest='output', help="Directory to save models", default='./training_checkpoints')
parser.add_argument('-e', '--epochs', dest='epochs', help='Number of training epochs', default=100)
parser.add_argument('-t', '--tensorboard', dest='tensorboard', help='Output directory for tensorboard logs', default='logs/gradient_tape/')
parser.add_argument('-b', '--batch', dest='batches', help='Batch Size for training data', default=64)
parser.add_argument('-cl', '--clean_logs', dest='clean_logs', help='Delete old logs in tensorboard log directory', default='False')
parser.add_argument('-cc', '--clean_checkpoints', dest='clean_checkpoints', help='Delete old model checkpoints in checkpoint directory', default='False')
parser.add_argument('-f', '--force', dest='force', help='force deletion of old Tensorboard logs and old Model Checkpoints from the supplied directory', default='False')

#parse CLI arguments
args = parser.parse_args()

#command line arguement inputs - See "help" description in section above for each of the following
img_dir = args.input_
checkpoint_dir = args.output
tensorboard_dir = args.tensorboard
epochs = int(args.epochs)
batches = int(args.batches)
clean_logs = args.clean_logs
clean_ckpts = args.clean_checkpoints
force_clean = args.force


'''Directory Cleanup Logic Section -- Used to Delete old Tensorboard logs and Model Checkpoints if commands passed from CLI to do so'''
#Check for forced cleanup - No warning prompt given, deletion will happen immediately
if force_clean.lower() == 'true':
    response = 'y'
    force='true'
    clean_logs ='true'
    clean_ckpts='true'
else:
    force=None
    response = None

#Warning prompt for cleaning up Tensorboard Logs
if clean_logs.lower() == 'true':
    while response not in ('y', 'n'):
        response = input('Warning: You have chosen to permanently delete files in the Tensorboard Log Directory. Do you wish to continue? [y/n]')
    if response == 'y':
        try:
            print('Cleaning Tensorboard Log Directory')
            shutil.rmtree(tensorboard_dir)
            print('Log Directory Cleared')
            response=None
        except Exception as e:
            print(e)
            response=None
    else:
        sys.exit()

#Warning prompt for cleaning up Model Checkpoints
if (clean_ckpts.lower() == 'true') | (force == 'true') :
    if force !='true':
        while response not in ('y', 'n'):
            response = input('Warning: You have chosen to permanently delete model checkpoints in the Checkpoint directory. Do you wish to continue? [y/n]')
        if response == 'y':
            try:
                print('Cleaning Checkpoint Directory')
                shutil.rmtree(checkpoint_dir)
                print('Checkpoint Directory Cleared')
            except Exception as e:
                print(e)
            else:
                sys.exit()
    else:
        try:
            print('Cleaning Checkpoint Directory')
            shutil.rmtree(checkpoint_dir)
            print('Checkpoint Directory Cleared')
        except Exception as e:
            print(e)
            


'''Data Input/Pipeline and Model Section'''
#input pipeline

#train_path = os.path.join(args.input_, "train")
train_dataset = make_dataset(args.input_)

#extract a test image to be logged to tensorboard during training
test = train_dataset.take(1)
for test_img, y, clss in test:
    test_img=test_img.numpy()
#test_img returns a dataset of batch size - extract the first image in the batch to be the test image
#test_img = test_img


'''Declare Model, Optimizer, and Metrics Section'''
#AutoEncoder model
model = XrayAE_Functional()

#Declare optimizer
optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

#Declare loss metrics
mse_loss_fn = tf.keras.losses.MAE
loss_metric = tf.keras.metrics.Mean('train_loss')


'''Admin Section'''
#Tensorboard logging
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_dir = tensorboard_dir + current_time + '/train'
#test_log_dir = tensorboard_dir + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
#test_summary_writer = tf.summary.create_file_writer(test_log_dir)

#Model Checkpoint writer
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)



'''Model Training Section'''
#Train model
fit(model, optimizer, epochs, train_dataset, train_dataset)




