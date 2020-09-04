# acoustic-images-self-supervision
Code for the paper "Leveraging Acoustic Images for Effective Self-Supervised Audio Representation Learning" ECCV 2020

## Requirements

- Python 3.7
- TensorFlow 1.14.0 >=

## Contents

The code is organized in several folders and there is a main script as follows:

- The `dataloader` folder contains files for loading the datasets.

- The `logger` folder contains file for tensorboard logger.

- The `models` folder contains our models.

- The `script` folder contains our bash scripts to perform experiments.

- The `trainer` folder contains code used to train our models.

- The `main` script is used for training and testing the models both in self-supervised and supervised way.

![work in progress](https://cdn5.vectorstock.com/i/1000x1000/90/79/under-construction-icon-on-white-background-under-vector-19719079.jpg)

## Preparing the dataset

For researchers who wish to use ACoustic Images and Videos in the Wild dataset for non-commercial research and/or educational purposes, please write to valentina.sanguineti@iit.it an email with your name, organization explaining the research purpose for which you would like to utilize the dataset and we will provide a link to download it.

The dataset is delivered as compressed zip files.
Once downloaded and decompressed, the data has to be converted into TensorFlow's native
[TFRecord](https://www.tensorflow.org/api_docs/python/python_io#tfrecords-format-details) format. Each TFRecord
will contain a [TF-SequenceExample](https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/core/example/example.proto)
protocol buffer with 1 second of data from the three modalities, video images, raw audio, and acoustic images.
It also contains labels data for video number and class.

Using `convert_data.py` script you have to provide the location of the
dataset and the folder where you want to save TFRecords. How to
generate TFRecords for all modalities:

```shell
# location where dataset was decompressed
ROOT_RAW_DIR= folder where dataset is, containing all classes:
├── class_0
│   ├── data_000
│   │   ├── audio
│   │   ├── MFCC_Image
│   │   ├── video

# location where to save TFRECORDS
OUT_DIR=folder where TFRecords will be saved, containing all classes 
├── class_0
│   ├── data_000
│   │   ├── Data_001.tfrecord
# generate TFRecords for all modalities
python3 convert_data.py $ROOT_RAW_DIR $OUT_DIR --modalities 0 1 2
```

When the script has finished running, you will find several TFRecord files created in OUT_DIR folder. These files represent the full dataset sharded over 10 classes. The mapping from
labels to class names is as follows:

```
class_0: train
class_1: boat
class_2: drone
class_3: fountain
class_4: drill
class_5: razor
class_6: hair dryer
class_7: hoover
class_8: shopping cart
class_9: traffic
```
To know the length of each video so we use the `framecount.py` utility script to generate a txt file for each video containing its length in seconds. It can be used both for measuring the length of video or TFRecords video, passing the location of the
dataset or of the folder where you have saved TFRecords, in addition with a flag 0 for normal videos or 1 for TFRecords. Run the following:

```shell
python3 framecount.py $ROOT_RAW_DIR 0
python3 framecount.py $OUT_DIR 1

```

To split the dataset in training, validation and test we use the lists of TFRecords provided in the `lists` folder. 



