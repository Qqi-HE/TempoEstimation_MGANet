import tensorflow.keras.backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Lambda, Conv2D, Conv2DTranspose, Dense, BatchNormalization, 
    GlobalAveragePooling2D, AveragePooling2D, 
    Multiply, Reshape, Softmax, Flatten, Concatenate, Dot
)


def GroupedAttention(x, pooling_size, group_number, reduction=4):
    height = K.int_shape(x)[-3]
    length = K.int_shape(x)[-2]
    channel = K.int_shape(x)[-1]

    x_list = []
    f_seg = height // group_number
    for i in range(group_number):
        # context modeling
        x_i = Lambda(lambda x: x[:, i*f_seg: (i+1)*f_seg, :, :])(x)
        w_k = Conv2D(1, (1, 1))(x_i)
        w_k = Reshape((1, f_seg * length))(w_k)
        w_k = Softmax(axis=-1)(w_k)
        x_i = Reshape((f_seg*length, channel))(x_i)
        x_i = Dot(axes=(1, 2))([x_i, w_k]) # (F*T, C), (1, F*T) -> (1, C)

        # transform
        x_i = Flatten()(x_i)
        x_i = Dense(channel // reduction, activation='elu')(x_i)
        x_i = BatchNormalization()(x_i)
        x_i = Dense(channel, activation='sigmoid')(x_i)
        x_i = Reshape((1, channel))(x_i)
        for j in range(height // group_number // pooling_size):
            x_list.append(x_i)

    if len(x_list) > 1:
        x = Concatenate(axis=-2)(x_list)
    else:
        x = x_list[0]
    x = Reshape((height // pooling_size, 1, channel))(x)
    return x


def GAModule(x, channel_number, pooling_size, group_number):
    # Adjust channel number
    x = Conv2D(channel_number, (1, 1), padding='same', activation='elu')(x)
    x = BatchNormalization()(x)

    # Trunk branch
    x_t = x
    for i in range(3):
        x_t = Conv2D(channel_number, (3, 3), padding='same', activation='elu')(x_t)
        x_t = BatchNormalization()(x_t)
    x_t = AveragePooling2D((pooling_size, 1))(x_t)

    # Attention branchs
    x_a = GroupedAttention(x, pooling_size, group_number)

    # Fusion
    x = Multiply()([x_t, x_a])
    return x


def MultiScaleStage(x0, x1, x2, channel_number, pooling_size, group_number):
    # GAModules
    x0 = GAModule(x0, channel_number, pooling_size, group_number)
    x1 = GAModule(x1, channel_number, pooling_size, group_number)
    x2 = GAModule(x2, channel_number, pooling_size, group_number)

    # Exchange information
    u_sample = [Conv2DTranspose(channel_number, (1, 3), strides=(1, 2), padding='same', output_padding=(0, 1)) for i in range(3)]
    d_sample = [AveragePooling2D((1, 2)) for i in range(3)]
    x0_1 = d_sample[0](x0)
    x0_2 = d_sample[1](x0_1)
    x1_0 = u_sample[0](x1)
    x1_2 = d_sample[2](x1)
    x2_1 = u_sample[1](x2)
    x2_0 = u_sample[2](x2_1)
    x0_c = Concatenate()([x0, x1_0, x2_0])
    x1_c = Concatenate()([x0_1, x1, x2_1])
    x2_c = Concatenate()([x0_2, x1_2, x2])
    x0_c = Conv2D(channel_number, (1, 1), padding='same', activation='elu')(x0_c)
    x1_c = Conv2D(channel_number, (1, 1), padding='same', activation='elu')(x1_c)
    x2_c = Conv2D(channel_number, (1, 1), padding='same', activation='elu')(x2_c)
    x0_c = BatchNormalization()(x0_c)
    x1_c = BatchNormalization()(x1_c)
    x2_c = BatchNormalization()(x2_c)

    return x0_c, x1_c, x2_c


def MultiScale(x):
    x0 = x # (81, 256)
    x1 = AveragePooling2D((1, 2))(x0) # (81, 128)
    x2 = AveragePooling2D((1, 2))(x1) # (81, 64)

    # Stage 1 (81, T, 1) -> (27, T, 32)
    x0, x1, x2 = MultiScaleStage(x0, x1, x2, 32, 3, 3)
    # Stage 2 (27, T, 32) -> (9, T, 64)
    x0, x1, x2 = MultiScaleStage(x0, x1, x2, 64, 3, 3)
    # Stage 3 (9, T, 64) -> (3, T, 128)
    x0, x1, x2 = MultiScaleStage(x0, x1, x2, 128, 3, 3)
    # Stage 4 (3, T, 128) -> (1, T, 256)
    x0, x1, x2 = MultiScaleStage(x0, x1, x2, 256, 3, 1)

    return [x0, x1, x2]


def create_model(input_shape=(81, 128, 1), output_dim=256):
    inputs = Input(shape=input_shape)

    # Multi-scale
    x = BatchNormalization()(inputs)
    # x = inputs
    x = MultiScale(x)

    # Classifier
    for i in range(3):
        x[i] = Conv2D(256, (1, 3), padding='same', activation='elu')(x[i])
        x[i] = BatchNormalization()(x[i])
        x[i] = GlobalAveragePooling2D()(x[i])
    x = Concatenate()(x)
    x = Dense(output_dim)(x)
    x = Softmax()(x)

    return Model(inputs=inputs, outputs=x)