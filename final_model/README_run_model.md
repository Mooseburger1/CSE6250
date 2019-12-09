# Code of Conduct Model
This script restores 14 pretriained AutoEncoder models, InceptionResNetV2, and a pretrained Fully Connected layer. It then compiles the model into one high level model. It then will deserialize the data which rests in .tfrecords format and predict on their inputs. The outputs will be saved to CSV in the specified output directory. There are models save as .h5 files that will be automatically parsed and restored for you. It takes a few moments to set the model up upon script execution. Note, that this is the full model with all parameters, this is however NOT all the image data (.tfrecords files). Due to size limits, only a small subset of data is being provided to predict on. The total dataset in its entirety was in upwards of 1.6TB. Also due to the size of the model itself, some users computers might not be conducive to run the full image data. These models were trained with several EC2 instances on AWS with massive GPUs.

## Requirements
* tensorflow 2.0 or tensorflow 2.0-gpu - If you install GPU version, you are responsible for configuring CUDA and CUDNN 
```
pip install tensorflow
```
or 
```
pip install tensorflow-gpu
```

* pandas
```
pip install pandas
```
* numpy
```
pip install numpy
```
* matplotlib
```
pip install matplotlib
```


## Usage
python run_model.py -i tfrecords -m models -o . -b 10

params
-----

[-i, --input_directory] - full directory path to tfrecords files
[-m, --models_directory] - full directory path to saved .h5 models
[-o, --output_directory] - directory to output for csv file of predictions
[-b, --batch] - batch size for the data to be loaded into the model


## Data
The input data to this model is Tensorflow TFRecords. The binary files are automatically unmarshalled by the script to be processed through the model. Each .tfrecords file contains multiple instance of images. One observation is
1. Image
2. One hot encoded label
3. Class name

There are 14 classes in total, hence the one hot encoded labels are length 14

## Note
You will see the message:

```
WARNING:tensorflow: No training configuration found in save file: the model was *not* compiled. Compile it manually
```

You can ignore this warning 

Also, as noted in our video & paper, accuracy will be low due to not enough training time. It took almost 2 weeks just to train 14 autoencoders. The final model itself on the full 1.6TB datast took 24 hours to complete just one Epcoh. Due to time and money limitations, we had to stop training the final model early before it could converge.