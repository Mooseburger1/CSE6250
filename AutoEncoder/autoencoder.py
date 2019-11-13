import tensorflow as tf
import numpy as np
import argparse
from CheXpert import XrayAE
import cv2 as cv
import matplotlib.pyplot as plt
import os
import time
import datetime
from pipeline import get_data
import matplotlib.pyplot as plt
import showit




#training function
@tf.function
def train_step(x_train, y_train):
    with tf.GradientTape() as tape:
        reconstruction = model(x_train, training=True)
        loss = mse_loss_fn(y_train, reconstruction)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    loss_metric(loss)
        
#fit function       
def fit(model, optimizer, epochs, train, test):
    print('\n\nTraining Starting @ {}'.format(datetime.datetime.now()))
    for epoch in range(epochs):


        for (x_train, y_train) in train:
            train_step(x_train, y_train)

        

        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            
            test_y = model(test_img)


            showit.image(np.squeeze(test_img))
            plt.title('Original')
            plt.show()

            showit.image(np.squeeze(test_y))
            plt.title('Reconstructed')
            plt.show()

        if epoch % 10 == 0:
            print('Epoch [%s]: mean loss [%s]' % (epoch, loss_metric.result().numpy()))
        
        loss_metric.reset_states()



parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_directory', dest='input_', help="Directory location of training images", default='.')
parser.add_argument('-o', '--output_directory', dest='output', help="Directory to save models", default='./training_checkpoints')
parser.add_argument('-e', '--epochs', dest='epochs', help='Number of training epochs', default=100)
parser.add_argument('-t', '--tensorboard', dest='tensorboard', help='Output directory for tensorboard logs', default='logs/gradient_tape/')
parser.add_argument('-b', '--batch', dest='batches', help='Batch Size for training data', default=64)

args = parser.parse_args()


#command line arguement inputs - See "help" description in section above for each of the following
img_dir = args.input_
checkpoint_dir = args.output
tensorboard_dir = args.tensorboard
epochs = args.epochs
batches = int(args.batches)

'''Data Input/Pipeline and Model Section'''
#input pipeline
train_dataset = get_data(img_dir, BATCH_SIZE=batches)

test = train_dataset.take(1)
for test_img, y in test:
    test_img=test_img.numpy()



#AutoEncoder model
model = XrayAE(input_shape=[None,None,3], use_skip_connections=False)

#Declare optimizer
optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

#Declare loss metrics
mse_loss_fn = tf.keras.losses.MeanSquaredError()
loss_metric = tf.keras.metrics.Mean()

'''Admin Section'''
#Tensorboard logging
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_dir = tensorboard_dir + current_time + '/train'
test_log_dir = tensorboard_dir + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

#Model Checkpoint writer
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)


'''Model Training Section'''
#Train model
fit(model, optimizer, epochs, train_dataset, train_dataset)

