## Overall parameters
SR = 11025
N_FFT = 1024
N_HOP = 512
N_MEL = 81
N_FRAME = 128

N_CLASS  = 256
L_OFFSET = 30

SHAPE = (N_MEL, N_FRAME, 1)
BATCH_SIZE = 32
LR = 0.001

STEPS_PER_EPOCH = 500
N_EPOCH  = 1000
PATIENCE = 100


## Train data paths
raw_data_path   = 'E:\\Lab\\TempoEstimation\\Datasets\\data\\'
feature_path    = 'E:\\Lab\\TempoEstimation\\Datasets\\feature\\feature1025\\'
annotation_path = 'E:\\Lab\\TempoEstimation\\Datasets\\label\\new210422\\'

train_annotation_file = annotation_path + 'BTrain1107_train.tsv'
valid_annotation_file = annotation_path + 'BTrain1107_valid.tsv'
train_feature_file    = feature_path + 'BTrain1107.joblib',


## Test data paths
test_annotation_files = [
    annotation_path + 'ACMMirum.tsv',
    annotation_path + 'Hainsworth.tsv',
    annotation_path + 'GTzan.tsv',
    annotation_path + 'SMC.tsv',
    annotation_path + 'GiantSteps.tsv',
    annotation_path + 'Ballroom.tsv',
    annotation_path + 'ISMIR04.tsv',
]
test_feature_files = [
    feature_path + 'ACMMirum.joblib',
    feature_path + 'Hainsworth.joblib',
    feature_path + 'GTzan.joblib',
    feature_path + 'SMC.joblib',
    feature_path + 'GiantSteps.joblib',
    feature_path + 'Ballroom.joblib',
    feature_path + 'ISMIR04.joblib',
]