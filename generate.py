import math
import random
import numpy as np

from keras.preprocessing.image import apply_affine_transform
from keras.utils import to_categorical
from utils import std_normalize


def radom_crop(feature, label, label_min, label_max, augmentation, n_frame):
    if augmentation == True:
        scale_factor = random.choice([x / 100.0 for x in range(80, 124, 4)])
    else:
        scale_factor = 1
    
    # Rescale label
    label_scaled = label * scale_factor
    while round(label_scaled) >= label_max:
        scale_factor -= 0.04
        label_scaled = label * scale_factor
    while round(label_scaled) < label_min:
        scale_factor += 0.04
        label_scaled = label * scale_factor

    # Crop feature
    length = max(n_frame, math.ceil(n_frame * scale_factor))
    if feature.shape[1] < length:
        feature = np.pad(feature, ((0, 0), (0, length - feature.shape[1]), (0, 0)))
    offset = random.randint(0, feature.shape[1] - length)
    feature_cropped = feature[:, offset : offset + length, :]

    # Rescale feature
    if scale_factor != 1:
        feature_scaled = apply_affine_transform(
            feature_cropped, zy=scale_factor, fill_mode='constant'
        ).astype(np.float32)
        if scale_factor > 1:
            begin = int(length/2 * (1 - 1/scale_factor))
        else:
            begin = 0
        feature_scaled = feature_scaled[:, begin: begin+n_frame, :]
    else:
        feature_scaled = feature_cropped
    
    return feature_scaled, label_scaled


def scaled_mel_spectrograms(features, labels, augmentation, label_offset, n_class, n_frame, batch_size):
    keys = list(labels.keys())
    random.shuffle(keys)
    i = 0

    x, y = [], []
    while True:
        i += 1
        if i >= len(keys):
            random.shuffle(keys)
            i = 0
        feature = features[keys[i]]
        label   = labels[keys[i]]

        # Augmentation
        feature_scaled, label_scaled = radom_crop(
            feature, label, label_offset, label_offset+n_class, augmentation, n_frame
        )
        
        # Append to the batch
        x.append(std_normalize(feature_scaled))
        y.append(round(label_scaled - label_offset))
        if len(x) == batch_size:
            y_coded = to_categorical(y, num_classes=n_class)
            yield np.stack(x, axis=0), np.stack(y_coded, axis=0)
            x, y = [], []

