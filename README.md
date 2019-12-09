# CSE6250

## AutoEncoder Usage:
```
python autoencoder.py 
[-i, --input_directory] - Directory location of training images (Default = '.')
[-o, --output_directory] - Directory to save model checkpoints (Default = './training_checkpoints')
[-e, --epochs] - Number of training epochs (Default = 100)
[-t, --tensorboard] - Output directory of Tensorboard Logs (Default = './logs/gradient_tape/')
[-b, --batch] - Batch Size for training data (Default = 64)
[-cl, --clean_logs] - Delete old logs in tensorboard log directory {true, false} (Default = false)
[-cc, --clean_checkpoints] - Delete old model checkpoints in checkpoint directory {true, false} (Default = false)
[-f, --force] - Force clean both logs and checkpoints with no warning prompts {true, false} (Default = false)
```

### Data Structure
Tensorflow reads directories for data ingestion and infers class labels from the name of the directories themselves. To properly setup your directory for data pipelining, you must follow the given structure:
```
train_data
|--Cardiomegaly
|      |--img1.jpg
|      |--img2.jpg
|      |--img3.jpg
|--Enlarged_Cardiomediastinum
|      |--img1.jpg
|      |--img2.jpg
|      |--img3.jpg
|--Lung_Lesion
|      |--img1.jpg
|      |--img2.jpg
|      |--img3.jpg


.........


```    
The script **pipeline.py** supplies the function used in **autoencoder.py** to prepare the training dataset. It utilizes Tensorflow's Dataset API and batching. It currently doesn't use the most optimal pipeline of TFRecords. When calling the autoencoder script, the CLI parameters for the given example would be
```
python autoencoder.py -i train_data
```

## Chest X-ray Disease Diagnosis
----------------------------------

### Requirements
* Python 3.7
* Tensorflow 2.0
* CPU processing supported - GPU Recommended

### Cannoncial Directory structures
All modules require a cannoncial directory structre of:
* A parent directory
* Sub directories where directory names are the classes
* data corresponding to the upper level subdirectory

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

<img src='https://www.ebmconsult.com/content/images/Xrays/ChestXrayAPNmlLabeled.png'  >

* Mentor1: Rishabh Bhardwaj hrbhardwaj39@gatech.edui
* Mentor2: Su Young Park hspark638@gatech.edui

X-rays are the oldest and most frequently used form of
medical imaging, but they require significant training for
clinicians to read correctly. This makes the analysis of xrays costly, time consuming, and prone to error. Luckily, the
latest big data techniques, especially deep learning, are making
automated analysis of x-ray images increasingly more realistic,
and groups are publishing large x-ray imaging dataset to help
researchers train, test, and improve their approaches. Creating
an automated diagnosis system would speed up processing,
reduce effort from clinicians, reduce errors, and make xrays more practical for diagnoses that currently rely on more
expensive but easier to analyze technologies like computerized
tomography.
The goal of this project is to reproduce and improve
previous study or propose a new study using at least one of the
given dataset (see below). If you only use CheXpert data[12],
you might also want to paricipate in the CheXpert competition
to get bonus points according to your performance.

## Data and Information

[Project Description](https://d1b10bmlvqabco.cloudfront.net/attach/jxaghvsf2i16a2/hknv39pnzou3m8/k0vmtt8uap72/CSE6250_project_2019Fall.pdf)

[cheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)

[NIH](https://nihcc.app.box.com/v/ChestXray-NIHCC)

[JSRT](http://db.jsrt.or.jp/eng.php)


# Responsibilites
### Dane
<img src="https://media.licdn.com/dms/image/C4E03AQEIr99v4-G6lg/profile-displayphoto-shrink_200_200/0?e=1581552000&v=beta&t=mfTLIngW4ZUMU9b9f5VRjfStnT0JedxPvtpg_4AsXo0">

* Generate initial statistics over the raw data to make sure the data quality is good enough and the key assumption about the data are met
* Identify the high-level technical approaches for the project (e.g., what algorithms to use or pipelines to use).

------------------

### Josh
<img src="https://avatars0.githubusercontent.com/u/241967?s=460&v=4" height=200 width=200>

* Conduct literature search to understand the state of arts and the gap for solving the problem.
* Prepare a timeline and milestones of deliverables for the entire project. **SCRUM MASTER**
------------

### Scott
<img src="https://media.licdn.com/dms/image/C5103AQF9GDUxajWA0Q/profile-displayphoto-shrink_200_200/0?e=1581552000&v=beta&t=-aXEqS_1I-eXXTxfHR0V1G0W3BM5YIT7twk80fXHYos">

* Identify and motivate the problems that you want to address in your project.
* Formulate the data science problem in details (e.g., classification vs. predictive modeling vs. clustering problem). 
* Identify clearly the success metric that you would like to use (e.g., AUC, accuracy, recall, speedup in running time).

### Team
-----------
* Setup the analytic infrastructure for your project (including both hardware and software environment, e.g., Azure or local clusters with Spark, python and all necessary packages).
* Discover the key data that will be used in your project and make sure an efficient path for obtaining the dataset. This is a crucial step and can be quite time-consuming, so do it on the first day and never stops until the project completion.

