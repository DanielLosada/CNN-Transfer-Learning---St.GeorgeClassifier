# St.GeorgeClassifier
This project is an image classification task using Convolutional Neural Networks (CNNs) to classify images into two different categories. The categories are St. George, and Not St. George. The problem to solve is, given an image, detect if St. George it's in the image or not.

## Dataset
The dataset it's a two folder directory. The first folder contains images in which St. George appear. The second folder contains images where St. George does **not** appear.
The dataset can be downloaded here: https://drive.google.com/drive/folders/1fIHdM54Q_eN5ZxF5nAGMaVIirMlNf3Sk?usp=sharing

## Approach
I approached this image classification task using two models. The first one (called Smallnet) it's a simple model made by me. The second approach (called FeatureClassifier) it's using the VGG16 pretrained model to use it as a feature extractor for another model made. In both cases I followed this steps:
* Data preparation: I started by unzipping the original zip file. Then the images are ordered in three sub-folders (train: 70%, eval: 20%, test: 10%). In the Smallnet, I also calculated the mean and std of the dataset for normalization. In both cases, the dataset is loaded with some standarization transformations applied over the images. In the FeatureClassifier, I applied data augmentation techniques that consist of random transformations of the images of the training split.

* Model building: 
    * Smallnet it's a CNN model with several convolutional and pooling layers, followed by fully connected layers. I used the ReLU activation function and added dropout regularization to prevent overfitting. The final layer it's a Sigmoid function, as we are dealing with binary classfication. 
    * FeatureClassifier its the VGG16 net, where I freezed the first 23 layers, and I retrained the layers from 24 to 30. After that, I added some fully connected layers using as before ReLU and dropout. To get the final prediction I used Sigmoid. 


* Model training: I trained the models using early stop with patience 3. That means that if in 3 cosecutive epochs the validation loss it's not improving, we stop the training. As optimizer, I used Adam, adding weight_decay regularization to prevent overfitting. I used Binary Cross Entropy as loss function. I monitored the training and validation accuracy to prevent overfitting.

* Model evaluation: I tested the best models on the train test. It's possible to test it on single photos or on the whole test split.

## Installation
### With Conda
Create a conda environment by running
```
conda create --name StGeorgeClassifier python=3.8
```
Then, activate the environment
```
conda activate StGeorgeClassifier
```
and install the dependencies
```
pip install -r requirements.txt
```
The first line of the file
```
--extra-index-url https://download.pytorch.org/whl/cu117
```
it's used to install the torch version compatible with cuda in windows.


## Running the project

If it's the first time running the project, you will need to prepare the dataset. For that you need to have 'george_test_task.zip' in the root project folder. 
You can start by the Smallnet or the FeatureClassifer. In both cases, you can use the flag **--preparedataset**. This flag will only prepare the dataset. If you don't use any flag, the training will start, and it's assumed that the dataset is already prepared. I the Smallnet case, you can use also **--calcmeanstd**, that will only calculate the mean and std of the dataset. For that, it's also assumed that you already prepared the dataset. The project only accepts one flag at a time, so in case of willing to execute everything consecutively, use the flag **--all**. That will prepare the dataset, in the Smallnet will calculate the mean and std, and will start the training process.

To run the Smallnet use:
```
python main_smallnet.py
```

To run the FeatureClassifier use:
```
python main_feature_classifier.py
```
## Testing
### Smallnet
For testing the smallnet with the test dataset run:

```
python test_smallnet.py
```

If you want to test it with only one image run:
```
python test_smallnet.py --singleimg
```
An image will be requested automatically.

### FeatureClassifer
For testing the FeatureClassifer with the test dataset run:

```
python test_feature_classifier.py
```

If you want to test it with only one image run:
```
python test_feature_classifier.py --singleimg
```
An image will be requested automatically.


