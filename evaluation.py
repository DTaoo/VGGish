from __future__ import division

import sys

sys.path.append('/home/hudi/anaconda2/lib/python2.7/site-packages/h5py')
sys.path.append('/home/hudi/anaconda2/lib/python2.7/site-packages/Keras-2.0.6-py2.7.egg')


import numpy as np
from numpy.random import seed, randint
from scipy.io import wavfile
from sklearn import svm
import linecache

from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from vggish import VGGish
from preprocess_sound import preprocess_sound


def loading_data(files, sound_extractor):


    lines = linecache.getlines(files)
    sample_num = len(lines)
    seg_num = 60
    seg_len = 5  # 5s
    data = np.zeros((seg_num * sample_num, 496, 64, 1))
    label = np.zeros((seg_num * sample_num,))

    for i in range(len(lines)):
        sound_file = '/mount/hudi/moe/sound_dataset/dcase/' + lines[i][:-7]
        sr, wav_data = wavfile.read(sound_file)

        length = sr * seg_len           # 5s segment
        range_high = len(wav_data) - length
        seed(1)  # for consistency and replication
        random_start = randint(range_high, size=seg_num)

        for j in range(seg_num):
            cur_wav = wav_data[random_start[j]:random_start[j] + length]
            cur_wav = cur_wav / 32768.0
            cur_spectro = preprocess_sound(cur_wav, sr)
            cur_spectro = np.expand_dims(cur_spectro, 3)
            data[i * seg_num + j, :, :, :] = cur_spectro
            label[i * seg_num + j] = lines[i][-2]

    data = sound_extractor.predict(data)

    return data, label


if __name__ == '__main__':

    sound_model = VGGish(include_top=False, load_weights=False)

    x = sound_model.get_layer(name="conv4/conv4_2").output
    output_layer = GlobalAveragePooling2D()(x)
    sound_extractor = Model(input=sound_model.input, output=output_layer)

    # load training data
    print "loading training data..."
    training_file = '/mount/hudi/moe/soundnet/train.txt'
    training_data, training_label = loading_data(training_file, sound_extractor)

    # load testing data
    print "loading testing data..."
    testing_file = '/mount/hudi/moe/soundnet/test.txt'
    testing_lines = linecache.getlines(testing_file)
    testing_data, testing_label = loading_data(testing_file, sound_extractor)

    clf = svm.LinearSVC(C=0.01, dual=False)
    clf.fit(training_data, training_label.ravel())
    p_vals = clf.decision_function(testing_data)

    test_count = len(testing_lines)
    pred_labels = np.zeros((test_count,))
    gt = testing_label[0:6000:60]
    p_vals = np.asarray(p_vals)

    for ii in range(test_count):
        scores = np.mean(p_vals[ii * 60:(ii + 1) * 60, :], axis=0)
        ind = np.argmax(scores)
        pred_labels[ii] = ind
    scores = gt == pred_labels
    score = np.mean(scores)
    print "accuracy: %f" % score



