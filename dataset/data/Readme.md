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
