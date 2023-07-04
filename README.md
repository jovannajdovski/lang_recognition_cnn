# Spoken Language Recognition Using Convolutional Neural Networks
This project shows how to train a language recognizer from scratch that is able to distinguish between Serbian, Spanish and English. It uses a dataset of audio recordings from [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets) for Serbian, English and Spanish languages, as well as various audio books. InceptionV3 and 5x-Conv-MaxPool models are trained in 60 epochs on a dataset of 30000 short audio recordings. With the data pre-processing, data augmentation and model the accuracy is 96.3 % for InceptionV3 and 92.4 % for 5x-Conv-MaxPool. 

### System Requirements
A fast CPU is recommended for data augmentation and pre-processing. For the model training, a CUDA GPU is necessary, especially for InceptionV3. You can check GPU CUDA capability [here](https://developer.nvidia.com/cuda-gpus). I trained the model with an Nvidia Geforce GTX 1650. The dataset coming from Mozilla Common Voice has a huge size. It might take a lot of time to process all of the data. 

### How to run
Before running the project you need to create a conda environment for tensorflow GPU capability by following the step-by-step instructions from [here](https://www.tensorflow.org/install/pip).


#### After that, create a conda environment:
```Bash
conda activate name_of_env
```

#### Install packages from a requirements file:
```Bash
pip install -r requirements.txt
```

#### After inserting the data and changing the paths in the modules, run python scripts in order:
```Bash
python prepare_dataset.py
```
```Bash
python augmentation.py
```
```Bash
python preprocessing.py
```

#### After that you can train InceptionV3 model with:
```Bash
python training.py
```

#### Or train 5x-Conv-MaxPool model with:
```Bash
python 5x-Conv-MaxPool_training.py
```

#### To evaluate the model run:
```Bash
python evaluation.py
```

#### Optionally you can see training details using TensorBoard by starting the server with
```Bash
tensorboard dev upload --logdir board
```
