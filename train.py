import os
import argparse
import importlib
import time

from tensorflow.keras.optimizers import Adam
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import load, generate, evaluation
from constants import *


##--- Prepare ---##
parser = argparse.ArgumentParser()
parser.add_argument('-n', "--network")
parser.add_argument('-m', "--mark")
args = parser.parse_args()

# Import model structure
network_module = importlib.import_module('network.' + args.network)
create_model   = network_module.create_model

# Checkpoint file
checkpoint = args.network + '_' + args.mark
checkpoint_file = f'checkpoints/{checkpoint}.h5'

# Log file
log_file = open(f'logs/{checkpoint}.log', 'wb')
def log(message):
    message_bytes = '{}\n'.format(message).encode(encoding='utf-8')
    log_file.write(message_bytes)
    print(message)


##--- Load data ---##
train_labels = load.labels(train_annotation_file)
log('Loaded {} training labels from {}.'.format(len(train_labels), train_annotation_file))
valid_labels = load.labels(valid_annotation_file)
log('Loaded {} validation labels from {}.'.format(len(valid_labels), valid_annotation_file))
train_features = load.features(train_feature_file)
log('Loaded features for {} files from {}.'.format(len(train_features), train_feature_file))


##--- Data Generator ---##
data_generator = generate.scaled_mel_spectrograms(
    train_features, train_labels, 
    augmentation=True, label_offset=L_OFFSET, n_class=N_CLASS, 
    n_frame=N_FRAME, batch_size=BATCH_SIZE
)


##--- Create model ---##
model = create_model(input_shape=SHAPE, output_dim=N_CLASS)
model.compile(loss='categorical_crossentropy', optimizer=(Adam(lr=LR)), metrics=['accuracy'])
model.summary()


##--- Training ---##
log('\nTaining...')
log('Model params: {}'.format(model.count_params()))

best_acc, best_epoch = 0, 0
for i_epoch in range(N_EPOCH):
    ## One epoch
    time_start = time.time()
    mean_loss, mean_acc = 0, 0
    for i_iter in range(STEPS_PER_EPOCH):
        # Train on one batch
        x, y = next(data_generator)
        batch_loss, batch_acc = model.train_on_batch(x, y)
        mean_loss += batch_loss / STEPS_PER_EPOCH
        mean_acc  += batch_acc  / STEPS_PER_EPOCH
        # Output
        print('Epoch {}/{} - {}/{} - loss {:.4f} - accuracy {:.4f}'.format(
            i_epoch + 1, N_EPOCH, i_iter + 1, STEPS_PER_EPOCH, batch_loss, batch_acc
        ), end='\r')

    # After one epoch
    time_train = time.time() - time_start
    log('Epoch {}/{} - time {:.1f}s - loss {:.4f} - accuracy {:.4f}     '.format(
        i_epoch + 1, N_EPOCH, time_train, mean_loss, mean_acc
    ))

    ## Validation
    predictions  = evaluation.get_predictions(model, SHAPE, train_features, valid_labels, L_OFFSET)
    valid_result = evaluation.report_accuracy_stats(predictions, valid_labels)

    # Renew check point
    if valid_result[1] >= best_acc:
        best_acc   = valid_result[1]
        best_epoch = i_epoch + 1
        model.save_weights(checkpoint_file)
    
    # Log validation results
    log('ACC0 {:.2f}%  ACC1 {:.2f}%  ACC2 {:.2f}%  Best {:.2f}% {}'.format(
        valid_result[0]*100, valid_result[1]*100, valid_result[2]*100, best_acc*100, 
        '☆' if best_epoch == i_epoch + 1 else f'({best_epoch})'
    ))

    # Early stopping
    if i_epoch - best_epoch > PATIENCE:
        log('Reach patience.')
        break

log_file.close()
