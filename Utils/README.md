## Note
* **make_tfrecords_single_classification.py** is used to convert a single folder of images into tfrecords files
* **make_tfrecords.py** is used to convert an entire directory of multiple image classifications into tfrecords files

## Usage
```
python make_tfrecords_single_classification.py
[-i, --input_directory] - Directory of training images
[-o, --output_directory] - Full path to the drectory to write sharded tfrecord files plus prefix name 
[-s, --shards] - Number of sharded tfrecord files to create
```

The output directory should contain the full path to directory you want to save the tfrecords plus the prefix of the file names. See example usage below

## Directory Structure 
#### make_tfrecords_single_classification.py
```
Lung_Lesion
    |--img1.jpg
    |--img2.jpg
    |--img3.jpg
    |--img4.jpg

```

#### make_tfrecords.py
```

train_data
|--Cardiomegaly
|       |--img1.jpg
|       |--img2.jpg
|       |--img3.jpg
|--Enlarged_Cardiomediastinum
|       |--img1.jpg
|       |--img2.jpg
|       |--img3.jpg
|--Lung_Lesion
|       |--img1.jpg
|       |--img2.jpg
|       |--img3.jpg
|--Lung_Opacity
|       |--img1.jpg
|       |--img2.jpg
|       |--img3.jpg
|--No_Finding
|       |--img1.jpg
|       |--img2.jpg
|       |--img3.jpg
|--Support_Devices
|       |--img1.jpg
|       |--img2.jpg
|       |--img3.jpg


...........................
```

## Example Usage
```
python make_tfrecords_single_classification.py --input_directory train_data --output_directory tfrecords_files/pleural_effusion --shards 15
```
**output**
```
tfrecords
|--pleural_effusion_train_01of15.tfrecord
|--pleural_effusion_train_02of15.tfrecord
|--pleural_effusion_train_03of15.tfrecord
|--pleural_effusion_train_04of15.tfrecord
|--pleural_effusion_train_05of15.tfrecord
|--pleural_effusion_train_06of15.tfrecord
|--pleural_effusion_train_07of15.tfrecord
|--pleural_effusion_train_08of15.tfrecord
|--pleural_effusion_train_09of15.tfrecord
|--pleural_effusion_train_10of15.tfrecord
|--pleural_effusion_train_110f15.tfrecord
|--pleural_effusion_train_12of15.tfrecord
|--pleural_effusion_train_13of15.tfrecord
|--pleural_effusion_train_14of15.tfrecord
|--pleural_effusion_train_15of15.tfrecord

```
