from __future__ import division

import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report

from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from vggish import VGGish

from dataset.dataset_utils import read_dataset_filenames, TRAIN_SET, TEST_SET, VAL_SET
from vggish_inputs import wavfile_to_examples


def loading_data(files, labels, sound_extractor):

    ret_data = []
    ret_labels = []
    batch_size=32
    batch=[]

    for file_name, label in zip(files, labels):
        wav_data = wavfile_to_examples(file_name)
        if len(wav_data) != 1:
            continue
        ret_labels.append(label)
        batch.append(wav_data)
        if len(batch) == batch_size:
            batch = np.concatenate(batch, axis=0)
            features = sound_extractor.predict(np.expand_dims(batch,-1))
            ret_data.append(features)
            batch = []

    if len(batch) > 0:
        batch = np.concatenate(batch, axis=0)
        features = sound_extractor.predict(np.expand_dims(batch, -1))
        ret_data.append(features)

    ret_data = np.concatenate(ret_data, axis=0)
    ret_labels = np.array(ret_labels)

    assert len(ret_labels) == len(ret_data)

    return ret_data, ret_labels

if __name__ == '__main__':

    sound_model = VGGish(include_top=False, load_weights=False)

    x = sound_model.get_layer(name="conv4/conv4_2").output
    output_layer = GlobalAveragePooling2D()(x)
    sound_extractor = Model(input=sound_model.input, output=output_layer)

    # load training data
    dataset = read_dataset_filenames()

    print("loading training data...")
    X_train, y_train = loading_data(*dataset[TRAIN_SET], sound_extractor)

    # load testing data
    print("loading test data...")
    X_test, y_test = loading_data(*dataset[TEST_SET], sound_extractor)

    # load testing data
    print("loading validation data...")
    X_val, y_val = loading_data(*dataset[VAL_SET], sound_extractor)

    print('Training...')

    clf = svm.LinearSVC()
    clf.fit(X_train, y_train)

    print('Report for training')
    y_pred = clf.predict(X_train)
    print(classification_report(y_train, y_pred))

    print('Report for validation')
    y_pred = clf.predict(X_val)
    print(classification_report(y_val, y_pred))

    print('Report for testing')
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))



