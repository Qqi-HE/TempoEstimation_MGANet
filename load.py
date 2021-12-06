import csv
import joblib


def labels(file):
    labels = {}
    with open(file, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            labels[row[0]] = float(row[1])
    return labels


def features(files):
    features = {}
    for feature_file in files:
        features.update(joblib.load(feature_file))
    return features

