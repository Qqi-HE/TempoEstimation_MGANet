import argparse
import importlib
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import load_model

from constants import *
import load
import evaluation


def test(annotation_file, feature_files, model):
    print('\n=============='+annotation_file.split('/')[-1].replace('.tsv', '')+'==============')

    ##--- Load data ---##
    labels = load.labels(annotation_file)
    print('Loaded {} labels from {}.'.format(len(labels), annotation_file))
    features = load.features(feature_files)
    print('Loaded features for {} files from {}.'.format(len(features), feature_files))

    ##--- Testing---##
    print('\nTesting...')
    predictions = evaluation.get_predictions(model, SHAPE, features, labels, L_OFFSET)
    test_result = evaluation.report_accuracy_stats(predictions, labels)
    for i, acc in enumerate(test_result):
        print('Accuracy{}: {:.4f}'.format(i, acc*100))


# Prepare
parser = argparse.ArgumentParser()
parser.add_argument('-n', "--network")
parser.add_argument('-m', "--mark")
args = parser.parse_args()

# Import model structure
network_module = importlib.import_module('network.' + args.network)
create_model = network_module.create_model

# Load weights
checkpoint = args.network + '_' + args.mark
checkpoint_file = f'checkpoints/{checkpoint}.h5'
model = create_model(input_shape=SHAPE, output_dim=N_CLASS)
model.load_weights(checkpoint_file)

for i in range(len(test_annotation_files)):
    test(test_annotation_files[i], [test_feature_files[i]], model)
