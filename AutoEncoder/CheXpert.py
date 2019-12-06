import tensorflow as tf
import numpy as np


#Convolutional Downsampling block layer
class downsample(tf.keras.layers.Layer):
    def __init__(self, filters, size, strides,bias=False, padding='same', activation='leaky', apply_batchnorm=True, initializer=None, input_shape=None):
        super(downsample, self).__init__()
        '''
        This class is a convolutional block for the "encoding" side of an autoencoer. It's constituents are 
        Convolutional Layer -> Batch Normalization -> Activation

        inputs
        ------
        filters: The number of convolutional filters for the downsampling block. Also refers to the output channels e.g. [length, width, filters]
        size: kernel size for the convolutional filters
        strides: striding step size for kernels
        bias: Add bias term to convolutional layer
        padding: zero-pad the input image if padding = 'same'
        activation: activation function to be used
        apply_batchnorm: Batch Normalize the outputs of the convolutional layer before passing into activation function
        initializer: initializer object for weight intilization

        outputs
        -------
        downsample layer of object type tf.keras.layers.Layer
        '''
        
        #create a keras sequential model to add layers
        self.result = tf.keras.Sequential()
        #add convolutional layer
        if input_shape:
            self.result.add(
                tf.keras.layers.Conv2D(filters, size, strides, padding, kernel_initializer=initializer, use_bias=bias, input_shape=input_shape)
            )
        else:
            self.result.add(
                tf.keras.layers.Conv2D(filters, size, strides, padding, kernel_initializer=initializer, use_bias=bias)
            )

        #apply batch normaliztion if true
        if apply_batchnorm:
            self.result.add(tf.keras.layers.BatchNormalization())
        
        #add activation function
        if activation=='leaky':
            self.result.add(tf.keras.layers.LeakyReLU())
        elif activation=='relu':
            self.result.add(tf.keras.layers.ReLU())
        elif activation=='tanh':
            self.result.add(tf.keras.activations.tanh())
        else: pass
            
    def call(self, x):
        #feed x through downsampling block
        return self.result(x)
    
    
#Convolutional Upsampling block
#Conv2D -> BatchNorm -> Dropout -> Activation
class upsample(tf.keras.layers.Layer):
    def __init__(self,filters, size,strides,padding='same', bias=False,activation='relu', apply_dropout=False, initializer=None):
        super(upsample, self).__init__()
        '''
        This class is a convolutional block for the "decoding" side of an autoencoer. It's constituents are 
        Convolutional Transpose Layer -> Dropout -> Activation

        inputs
        ------
        filters: The number of convolutional filters for the downsampling block. Also refers to the output channels e.g. [length, width, filters]
        size: kernel size for the convolutional filters
        strides: striding step size for kernels
        bias: Add bias term to convolutional layer
        padding: zero-pad the input image if padding = 'same'
        activation: activation function to be used
        apply_dropout: Neuron dropout during training with 50% probability to prevent overfitting
        initializer: initializer object for weight intilization

        outputs
        -------
        upsample layer of object type tf.keras.layers.Layer
        '''
        #create a keras sequential model to add layers
        self.result=tf.keras.Sequential()
        #add convolutional transpose layer
        self.result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides, padding, kernel_initializer=initializer, use_bias=bias)
        )
        
        self.result.add(tf.keras.layers.BatchNormalization())
        
        if apply_dropout:
            self.result.add(tf.keras.layers.Dropout(0.5))
            
        if activation=='leaky':
            self.result.add(tf.keras.layers.LeakyReLU())
        elif activation=='relu':
            self.result.add(tf.keras.layers.ReLU())
        elif activation=='tanh':
            self.result.add(tf.keras.activations.tanh())
        else: pass
        
    def call(self, x):
        return self.result(x)
    
    
#Xray AutoEncoder Class
#Set 'use_skip_connections' to True to create U-Net AutoEncoder
class XrayAE(tf.keras.Model):
    def __init__(self, input_shape, use_skip_connections=True):
        super(XrayAE, self).__init__()
        '''
        This class creates the AutoEnocder model to be fit and trained. It inherts from tf.keras.Model and all associated methods with it. If setting 'use_skip_connections'
        to True, ensure that input image shape is one such that the output from each downsample layer is an even number. An odd output will throw an error in the upsample layers
        when skip connections are concatenated due to mismathing sizes.

        inputs
        ------
        use_skip_connctions: Set to True to create U-Net

        outputs
        -------
        tf.keras.Model object
        '''
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.skip_connections = use_skip_connections


        self.downsample = [
            downsample(filters=64,size=4,strides=2,bias=False, padding='same', activation='leaky', apply_batchnorm=False,initializer=self.initializer, input_shape=input_shape),
            downsample(filters=128,size=4,strides=2,bias=False, padding='same', activation='leaky', apply_batchnorm=True,initializer=self.initializer),
            downsample(filters=256,size=4,strides=2,bias=False, padding='same', activation='leaky', apply_batchnorm=True,initializer=self.initializer),
            downsample(filters=512,size=4,strides=2,bias=False, padding='same', activation='leaky', apply_batchnorm=True,initializer=self.initializer),
            downsample(filters=512,size=4,strides=2,bias=False, padding='same', activation='leaky', apply_batchnorm=True,initializer=self.initializer),
            downsample(filters=512,size=4,strides=2,bias=False, padding='same', activation='leaky', apply_batchnorm=True,initializer=self.initializer),
            downsample(filters=512,size=4,strides=2,bias=False, padding='same', activation='leaky', apply_batchnorm=True,initializer=self.initializer),
            downsample(filters=512,size=4,strides=2,bias=False, padding='same', activation='leaky', apply_batchnorm=True,initializer=self.initializer)
        ]
        
        self.upsample=[
            upsample(filters=512,size=4,strides=2,padding='same', bias=False,activation='relu', apply_dropout=True,initializer=self.initializer),
            upsample(filters=512,size=4,strides=2,padding='same', bias=False,activation='relu', apply_dropout=True,initializer=self.initializer),
            upsample(filters=512,size=4,strides=2,padding='same', bias=False,activation='relu', apply_dropout=True,initializer=self.initializer),
            upsample(filters=512,size=4,strides=2,padding='same', bias=False,activation='relu', apply_dropout=False,initializer=self.initializer),
            upsample(filters=256,size=4,strides=2,padding='same', bias=False,activation='relu', apply_dropout=False,initializer=self.initializer),
            upsample(filters=128,size=4,strides=2,padding='same', bias=False,activation='relu', apply_dropout=False,initializer=self.initializer),
            upsample(filters=64 ,size=4,strides=2,padding='same', bias=False,activation='relu', apply_dropout=False,initializer=self.initializer),
            
        ]
        
        self.output_ = tf.keras.layers.Conv2DTranspose(filters=3,kernel_size=4,strides=2,padding='same',kernel_initializer=self.initializer,activation='tanh')
    

    def call(self, x):
        '''
        call is automatically called by tensorflow backend during training
        '''
        
        #U-Net AutoEncoder
        if self.skip_connections:
            concat = tf.keras.layers.Concatenate()
            skips = []

            for layer in self.downsample:
                x = layer(x)

                skips.append(x)

            skips = reversed(skips[:-1])

            for layer, skip in zip(self.upsample,skips):
                x = layer(x)
                x = concat([x, skip])

            x = self.output_(x)

        #Vanilla AutoEncoder    
        else:

            for layer in self.downsample:
                x = layer(x)

            for layer in self.upsample:
                x=layer(x)
                
            x = self.output_(x)

        return x





def Downsample(filters, size, apply_maxpool=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_maxpool:
    result.add(tf.keras.layers.MaxPool2D(2,2, 'same'))

  result.add(tf.keras.layers.LeakyReLU())

  return result

def Upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def XrayAE_Functional():
  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(3, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)



  inputs = tf.keras.layers.Input(shape=[299,299,3])
  x = inputs

  
  x = Downsample(32,4,True)(x)
  x = Downsample(32,4,True)(x)
  x = Downsample(32,4,True)(x)
  x = Downsample(16,4,True)(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(1200, input_shape=(5776,))(x)
  x = tf.keras.layers.Dense(5776, input_shape=(1200,))(x)
  x = tf.keras.layers.Reshape((19,19,16))(x)
  x = Upsample(16,4,True)(x)
  x = Upsample(32,4,True)(x)
  x = Upsample(32,4,True)(x)
  x = last(x)
  x = tf.keras.layers.Cropping2D(cropping=((2,3), (2,3)), input_shape=(304,304,1))(x)

  return tf.keras.Model(inputs=inputs, outputs=x)




def Model1_(transfer_model, cheatsheet_generator):
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = tf.keras.layers.Input(shape=[299,299,3])
    
    transfer = tf.keras.Sequential([transfer_model])
    cheat = cheatsheet_generator()
    
    x_trans = transfer(inputs)
    x_trans = tf.keras.layers.Flatten()(x_trans)
    x_cheat = cheat(inputs)
    x_cheat = tf.keras.layers.Flatten()(x_cheat)
    
    x = tf.keras.layers.Concatenate()([x_trans, x_cheat])
    
    
    
    x = tf.keras.layers.Dense(units=100, 
                              activation='relu', 
                              use_bias=True, 
                              kernel_initializer=initializer, 
                              bias_initializer=initializer)(x)
    
    x = tf.keras.layers.Dense(units=50, 
                              activation='relu', 
                              use_bias=True, 
                              kernel_initializer=initializer, 
                              bias_initializer=initializer)(x)
    
    x = tf.keras.layers.Dense(units=50, 
                              activation='sigmoid', 
                              use_bias=True, 
                              kernel_initializer=initializer, 
                              bias_initializer=initializer)(x)
    
    x = tf.keras.layers.Dense(units=14, 
                              activation='softmax', 
                              use_bias=True, 
                              kernel_initializer=initializer, 
                              bias_initializer=initializer)(x)
    
    
    return tf.keras.Model(inputs=inputs, outputs=x)

def Model1():
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = tf.keras.layers.Input(shape=[299,299,3])
    
    
    x = tf.keras.layers.Dense(units=100, 
                              activation='relu', 
                              use_bias=True, 
                              kernel_initializer=initializer, 
                              bias_initializer=initializer)(inputs)
    
    x = tf.keras.layers.Dense(units=50, 
                              activation='relu', 
                              use_bias=True, 
                              kernel_initializer=initializer, 
                              bias_initializer=initializer)(x)
    
    x = tf.keras.layers.Dense(units=50, 
                              activation='sigmoid', 
                              use_bias=True, 
                              kernel_initializer=initializer, 
                              bias_initializer=initializer)(x)
    
    x = tf.keras.layers.Dense(units=14, 
                              activation='softmax', 
                              use_bias=True, 
                              kernel_initializer=initializer, 
                              bias_initializer=initializer)(x)
    
    
    return tf.keras.Model(inputs=inputs, outputs=x)

def Model2():
    pass
