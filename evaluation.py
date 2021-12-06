import numpy as np
from utils import std_normalize


def report_accuracy_stats(predictions, labels):

    def same_tempo(true_value, estimated_value, factor=1., tolerance=0.04):
        if tolerance is None or tolerance == 0.0:
            return round(estimated_value * factor) == round(true_value)
        else:
            return abs(estimated_value * factor - true_value) < true_value * tolerance
        
    acc0_sum = 0
    acc1_sum = 0
    acc2_sum = 0
    count = 0
    for key, label in labels.items():
        if key in predictions:
            predicted_label = predictions[key]

            acc0 = same_tempo(label, predicted_label, tolerance=0.0)
            acc1 = same_tempo(label, predicted_label)
            acc2 = acc1 or same_tempo(label, predicted_label, factor=2.) \
                    or same_tempo(label, predicted_label, factor=1. / 2.) \
                    or same_tempo(label, predicted_label, factor=3.) \
                    or same_tempo(label, predicted_label, factor=1. / 3.)

            if acc0:
                acc0_sum += 1
            if acc1:
                acc1_sum += 1
            if acc2:
                acc2_sum += 1

        else:
            print('No prediction for key {}'.format(key))

        count += 1
    acc0_result = acc0_sum / float(count)
    acc1_result = acc1_sum / float(count)
    acc2_result = acc2_sum / float(count)
    
    return [acc0_result, acc1_result, acc2_result]


def get_predictions(model, input_shape, features, labels, label_offset):
    results = {}
    for key, label in labels.items():
        feature = np.copy(features[key])
        fragments = []
        cnt = feature.shape[1] // input_shape[1]
        for i in range(cnt):
            feature_cropped = feature[:, input_shape[1]*i : input_shape[1]*(i+1), :]
            feature_normed  = std_normalize(feature_cropped)
            fragments.append(feature_normed)  
        x = np.array(fragments)
        
        predictions = model.predict(x, x.shape[0])
        predictions = np.sum(predictions, axis=0)
        results[key] = np.argmax(predictions) + label_offset

    return results
