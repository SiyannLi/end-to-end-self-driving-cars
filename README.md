# End-to-End-Learning-for-Self-Driving-Cars

Modified based on https://github.com/Nuclearstar/End-to-End-Learning-for-Self-Driving-Cars

tested on Ubuntu 22
## Dataset
You can download the dataset from [here](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) for Implementation 1.<br/>
You can download the dataset and extract into the repository folder from [here](https://github.com/SullyChen/driving-datasets) for Implementation 2.

1. Implementation 1
2. Implementation 2

## Code Requirements
pip install requirements.txt


## Python Implementation
1. Network Used- Convolutional Neural Network(CNN)
2. Keras-Tensorflow Architecture

### Procedure in Implementation 1
1. First use python LoadData.py which will get dataset from folder and store it in a pickle file.
2. Now you need to have the data, use python TrainModel.py which will load data from pickle and augment it. After this, the training process begins.
3. For testing it on the video you need to use python DriveApp.py

### Procedure in Implementation 2
1. Use python train.py to train the model.
2. Use python run.py to run the model on a live webcam feed.
3. Use python run_dataset.py to run the model on the dataset

## References and Credits:
1. Modified based on Nuclearstar's Repo
2. This implementation is inspired by Akshay Bahadur's project
3. This implementation also took a lot of inspiration from the Sully Chen's work.
