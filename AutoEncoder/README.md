## Note
* autoencoder3.py (version 3) is the most up to date version. 
* Input data is expected to be TFRecords file format - Run scripts in CSE6250/Utils to create TFRecords first
* The script **pipeline.py** supplies the function used in **autoencoder.py** to prepare the training dataset. It utilizes Tensorflow's Dataset API and batching. 

## AutoEncoder Usage:
```
python autoencoder3.py 
[-i, --input_directory] - Directory location of training images
[-o, --output_directory] - Directory to save model checkpoints
[-c, --class] - Specified class of images to train the AutoEncoder on
[-e, --epochs] - Number of training epochs (Default = 100)
[-t, --tensorboard] - Output directory of Tensorboard Logs
[-b, --batch] - Batch Size for training data (Default = 64)
[-cl, --clean_logs] - Delete old logs in tensorboard log directory {true, false} (Default = false)
[-cc, --clean_checkpoints] - Delete old model checkpoints in checkpoint directory {true, false} (Default = false)
[-f, --force] - Force clean both logs and checkpoints with no warning prompts {true, false} (Default = false)
```

### Input Data Structure
Tensorflow reads directories for data ingestion and infers class labels from the name of the directories themselves. To properly setup your directory for data pipelining, you must follow the given structure. Note that you only have to supply
the parent directory. The script will infer and load both the train data and valid data from the directory structure:
```
data
|--train_data
|       |--Cardiomegaly
|       |       |--img1.jpg
|       |       |--img2.jpg
|       |       |--img3.jpg
|       |--Enlarged_Cardiomediastinum
|       |       |--img1.jpg
|       |       |--img2.jpg
|       |       |--img3.jpg
|       |--Lung_Lesion
|       |       |--img1.jpg
|       |       |--img2.jpg
|       |       |--img3.jpg
|--valid_data
|       |--Cardiomegaly
|       |       |--img1.jpg
|       |       |--img2.jpg
|       |       |--img3.jpg
|       |--Enlarged_Cardiomediastinum
|       |       |--img1.jpg
|       |       |--img2.jpg
|       |       |--img3.jpg
|       |--Lung_Lesion
|       |       |--img1.jpg
|       |       |--img2.jpg
|       |       |--img3.jpg


...........................
```
### Output Directory Structure

While training, the program will save checkpoints of the tensorflow model as well as logs for Tensorboard. After completing the final Epoch, the program will save the entire model as a h5 file. You must preconstruct this output directory structure. Its path will be used as input as a command line parameter.
```
ex.


pneumonia
|--checkpoints
|--logs
|--model
```


## Example Usage
Example usage using directory structure **data** and **pneumonia** noted in above section

```
python autoencoder3.py --input_directory data --output_directory pneumonia --class pneumonia --epochs 200 --batch 64
```
The ```--class``` tag is required since the input directory is the path to the parent directory of all the data. The model needs to know specifically which class to read data from in the input directory. Again, both the Train and Valid data will be read for the specified class in the input directory


   


