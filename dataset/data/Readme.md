# Readme
This folder must contain a sub set of the 
[Speech Commands dataset](https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz)

Download and move a sub set (i.e. some folders) of the dataset before running the `evaluation.py`.

The directory should look like this:

```
VGGish/dataset/data/class_01/*.wav
VGGish/dataset/data/class_02/*.wav
VGGish/dataset/data/class_03/*.wav
```

where `class_01`, `class_02`, and `class_03` are classes from the Speech Commands dataset, 
and `VGGish` is the root of this project.

You can check that the dataset is correctly placed, and see a summary
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

Run `python dataset/dataset_utils.py --verbose` to see every single file considered for the dataset.
