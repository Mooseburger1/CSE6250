## Note
You must have AutoEncoders trained and saved as .h5 files before running this script
* train_final_model_v2.py is the most up to date version

## Usage
```
python train_final_model_v2.py
[-m, --models_directory] - Directory location of trained AutoEnocder Models
[-t, --tfrecords] - Directory location of tfrecords
[-o, --output_directory] - Directory to save all data
[-e, --epochs] - Number of training epochs
[-b, --batch] - Batch size for training data
[-n, --number_model_to_train] - Which architecture to train
```

**--number_model_to_train** specifies which architecture to train. Currently there are 2 different architectures available. To add more models, add the architecture to the script AutoEnocder/CheXpert.py

## Saved Models Directory
The models_directory input should be the high level directory of all the output folders you created from training the AutoEncoders. The script will parse through the directory and extract the .h5 files

```
autoencoders
|--pneumonia
|     |--checkpoints
|     |--logs
|     |--model
|          |--pneumonia_model.h5
|--support_device
|     |--checkpoints
|     |--logs
|     |--model
|         |--support_device.h5
|--lung_lesion
|     |--checkpoints
|     |--logs
|     |--model
|         |--lung_lesion.h5
```
### Output Directory Structure

While training, the program will save checkpoints of the tensorflow model as well as logs for Tensorboard. After completing the final Epoch, the program will save the entire model as a h5 file
```
ex.


final_model
|--checkpoints
|--logs
|--model
```

## Example Usage
```
python train_final_model_v2.py -m autoencoders --tfrecords saved_tfrecords --output_directory final_model --epochs 200 --batch 64 -n 1
```

