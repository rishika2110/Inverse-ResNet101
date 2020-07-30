import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Activation, Add, AveragePooling2D, BatchNormalization, Conv2D, Conv2DTranspose, Dense, Flatten, Input, Reshape, ZeroPadding2D

def dres_conv(x, s, filters):
    # here the input size changes
    x_skip = x
    f1, f2 = filters

    # third block
    x = Conv2DTranspose(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    # second block
    x = Conv2DTranspose(f1, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    # third block
    x = Conv2DTranspose(f1, kernel_size=(1, 1), strides=(s, s), padding='valid')(x)
    # when s = 2 then it is like downsizing the feature map
    x = BatchNormalization()(x)

    # shortcut 
    x_skip = Conv2DTranspose(f1, kernel_size=(1, 1), strides=(s, s), padding='valid')(x_skip)
    x_skip = BatchNormalization()(x_skip)

    # add 
    x = Add()([x, x_skip])
    x = Activation(activations.relu)(x)

    return x

def dres_identity(x, filters): 
    # dresnet block where dimension does not change.
    # The skip connection is just simple identity conncection
    # There will be 3 blocks and then input will be added

    x_skip = x # this will be used for addition with the residual block 
    f1, f2 = filters

    # first block 
    x = Conv2DTranspose(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)


    # second block # bottleneck (but size kept same with padding)
    x = Conv2DTranspose(f1, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    # third block activation used after adding the input
    x = Conv2DTranspose(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = BatchNormalization()(x)

    # add the input 
    x = Add()([x, x_skip])
    x = Activation(activations.relu)(x)

    return x