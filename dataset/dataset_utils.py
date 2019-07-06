import hashlib
import os
import re

import numpy as np

MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1  # ~134M
VAL_PERCENTAGE = 10
TEST_PERCENTAGE = 10

TRAIN_SET = 'training'
VAL_SET = 'validation'
TEST_SET = 'testing'

DATA_SET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')  # path to the dataset


def which_set(filename, validation_percentage, testing_percentage):
    """Determines which data partition the file should belong to.

    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.

    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.

    Args:
      filename: File path of the data sample.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.

    Returns:
      String, one of 'training', 'validation', or 'testing'.
    """
    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.

    hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8'))
    hash_name_hashed = hash_name_hashed.hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (MAX_NUM_WAVS_PER_CLASS + 1)) *
                       (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = VAL_SET
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = TEST_SET
    else:
        result = TRAIN_SET
    return result


def read_dataset_filenames():

    classes_names = [file_name for file_name in os.listdir(DATA_SET_PATH)
                     if os.path.isdir(os.path.join(DATA_SET_PATH, file_name))]

    dataset = {
        TRAIN_SET: ([], []),  # Pair <X, y> where X is the list of paths to the files and y the class names
        TEST_SET: ([], []),
        VAL_SET: ([], [])

    }

    for class_name in classes_names:
        for file_name in os.listdir(os.path.join(DATA_SET_PATH, class_name)):

            _x_path = os.path.join(DATA_SET_PATH, class_name, file_name)
            _set = which_set(_x_path, VAL_PERCENTAGE, TEST_PERCENTAGE)

            dataset[_set][0].append(_x_path)
            dataset[_set][1].append(class_name)

    return dataset


def main(verbose):

    dataset = read_dataset_filenames()

    if verbose:

        for k, (x, y) in dataset.items():
            assert len(x) == len(y)
            for _x, _y in zip(x, y):
                print(f'\t[{_y}] | {_x}')

    print('\tDataset summary:')

    for k, (x, y) in dataset.items():
        print(f'\n\t{k:<10} -> {len(x): 5d} examples')
        _classes, _cnts = np.unique(y, return_counts=True)
        for _class, _n in zip(_classes, _cnts):
            print(f'\t\t{_class:>5} -> {_n: 5d} examples')


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description=f'This script allows a fast check on the read_dataset_filenames'
    f' function and shows the samples that are being taken when calling it.')

    parser.add_argument("-v", "--verbose", action='store_true',
                        help="Show every single sample path")

    args = parser.parse_args()

    main(verbose=args.verbose)
