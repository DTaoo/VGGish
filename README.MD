# VGGish: A VGG-like audio classification model 

This repository provides a VGGish model, implemented in Keras with tensorflow backend. This repository is developed 
based on the model for [AudioSet](https://research.google.com/audioset/index.html). 
For more details, please visit the [slim version](https://github.com/tensorflow/models/tree/master/research/audioset).

## Pretrained weights in Keras h5py:

They should be downloaded into `weights` directory

* [Model](https://drive.google.com/open?id=1mhqXZ8CANgHyepum7N4yrjiyIg6qaMe6) with the top fully connected layers

* [Model](https://drive.google.com/open?id=16JrWEedwaZFVZYvn1woPKCuWx85Ghzkp) without the top fully connected layers

  
  
 ## Test it!
You can check it with a sub-set of the [Speech Commands dataset](https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz)
 
Download it and move a sub set (i.e. some folders) of the dataset.

The directory should look like this:

```
VGGish/dataset/data/class_01/*.wav
VGGish/dataset/data/class_02/*.wav
VGGish/dataset/data/class_03/*.wav
```

where `class_01`, `class_02`, and `class_03` are classes from the Speech Commands dataset, 
and `VGGish` is the root of this project. You can check that the dataset is correctly placed, and see a summary
 by running `python dataset/dataset_utils.py` which, for yes/no categories, should output something like that:

```
        Dataset summary:

        training   ->  6358 examples
                   no ->  3130 examples
                  yes ->  3228 examples

        testing    ->   824 examples
                   no ->   405 examples
                  yes ->   419 examples

        validation ->   803 examples
                   no ->   406 examples
                  yes ->   397 examples

```

Then you can run it with `python evaluation.py`.

Here you can see an output example for yes/no categories 

```
loading training data...
loading test data...
loading validation data...
Training...
Report for training
              precision    recall  f1-score   support

          no       0.88      0.90      0.89      2850
         yes       0.90      0.88      0.89      2990

    accuracy                           0.89      5840
   macro avg       0.89      0.89      0.89      5840
weighted avg       0.89      0.89      0.89      5840

Report for validation
              precision    recall  f1-score   support

          no       0.86      0.90      0.88       363
         yes       0.89      0.85      0.87       356

    accuracy                           0.87       719
   macro avg       0.87      0.87      0.87       719
weighted avg       0.87      0.87      0.87       719

Report for testing
              precision    recall  f1-score   support

          no       0.90      0.90      0.90       386
         yes       0.90      0.90      0.90       397

    accuracy                           0.90       783
   macro avg       0.90      0.90      0.90       783
weighted avg       0.90      0.90      0.90       783

```
 Notice that I removed some logging info from TF which is not relevant at this point.
 
 ## Reference:

* Gemmeke, J. et. al.,
  [AudioSet: An ontology and human-labelled dataset for audio events](https://research.google.com/pubs/pub45857.html),
  ICASSP 2017

* Hershey, S. et. al.,
  [CNN Architectures for Large-Scale Audio Classification](https://research.google.com/pubs/pub45611.html),
  ICASSP 2017
