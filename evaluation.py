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
    """
    Given a list of filenames and its laabels, extract a feature vector with the sound_extractor given
    :param files: List of path to .wav files
    :param labels: List of class names, correlative to the files list
    :param sound_extractor: Keras model used to compute the embeddings/features
    :return: List of features and list of labels
    """

    ret_data = []
    ret_labels = []
    batch_size = 32
    batch = []

    for file_name, label in zip(files, labels):
        # compute log mel spectogram from audio files
        log_mel = wavfile_to_examples(file_name)

        # TODO: not sure why for some files the output is empty, should look closer at this
        if len(log_mel) != 1:
            continue
        # add the label to the return list and the spectogram to the batch
        ret_labels.append(label)
        batch.append(log_mel)
        if len(batch) == batch_size:
            # when batch is full, run it through the net to get the embeddings/features
            batch = np.concatenate(batch, axis=0)
            features = sound_extractor.predict(np.expand_dims(batch,-1))
            ret_data.append(features)
            # reset the batch
            batch = []

    if len(batch) > 0:
        # check if there is data in the batch waiting to be run through the net
        batch = np.concatenate(batch, axis=0)
        features = sound_extractor.predict(np.expand_dims(batch, -1))
        ret_data.append(features)

    ret_data = np.concatenate(ret_data, axis=0)
    ret_labels = np.array(ret_labels)

    # check everything is as expected
    assert len(ret_labels) == len(ret_data)

    return ret_data, ret_labels


if __name__ == '__main__':

    # define the feature extractor
    sound_model = VGGish(include_top=False, load_weights=True)

    x = sound_model.get_layer(name="conv4/conv4_2").output
    output_layer = GlobalAveragePooling2D()(x)
    sound_extractor = Model(input=sound_model.input, output=output_layer)

    # load dataset
    dataset = read_dataset_filenames()

    # load the training data and compute its features/embeddings
    print("loading training data...")
    X_train, y_train = loading_data(*dataset[TRAIN_SET], sound_extractor)

    # load the testing data and compute its features/embeddings
    print("loading test data...")
    X_test, y_test = loading_data(*dataset[TEST_SET], sound_extractor)

    # load the validation data and compute its features/embeddings
    print("loading validation data...")
    X_val, y_val = loading_data(*dataset[VAL_SET], sound_extractor)

    print('Training...')
    # Train simple SVM classifier from sckit, using its default parameters
    clf = svm.LinearSVC()
    clf.fit(X_train, y_train)

    # Evaluate the model in all sets
    print('Report for training')
    y_pred = clf.predict(X_train)
    print(classification_report(y_train, y_pred))

    print('Report for validation')
    y_pred = clf.predict(X_val)
    print(classification_report(y_val, y_pred))

    print('Report for testing')
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))



