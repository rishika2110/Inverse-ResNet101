from layers import dres_conv, dres_identity
import tensorflow as tf
from tensorflow.keras.applications import ResNet101

def Encoder(shape):
    input_im = Input(shape=(shape[1], shape[2], shape[3]))
    Encoder = ResNet101(include_top=False, weights='imagenet', input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    x = Encoder(input_im)
    x = Flatten()(x)
    encoding = Dense(2048, kernel_initializer='he_normal')(x)
    return tf.keras.Model(inputs=input_im, outputs=encoding, name='Encoder')

def Decoder():
    dec_input = Input(shape=(2048,))
    x = Dense(2 * 2 * 2048, kernel_initializer='he_normal')(dec_input)
    x = Reshape((2, 2, 2048))(x)

    x = dres_conv(x, s=2, filters=(512, 2048))
    x = dres_identity(x, filters=(512, 2048))
    x = dres_identity(x, filters=(512, 2048))

    x = dres_conv(x, s=2, filters=(256, 1024))
    x = dres_identity(x, filters=(256, 1024))
    x = dres_identity(x, filters=(256, 1024))
    x = dres_identity(x, filters=(256, 1024))
    x = dres_identity(x, filters=(256, 1024))
    x = dres_identity(x, filters=(256, 1024))
    x = dres_identity(x, filters=(256, 1024))
    x = dres_identity(x, filters=(256, 1024))
    x = dres_identity(x, filters=(256, 1024))
    x = dres_identity(x, filters=(256, 1024))
    x = dres_identity(x, filters=(256, 1024))
    x = dres_identity(x, filters=(256, 1024))
    x = dres_identity(x, filters=(256, 1024))
    x = dres_identity(x, filters=(256, 1024))
    x = dres_identity(x, filters=(256, 1024))
    x = dres_identity(x, filters=(256, 1024))
    x = dres_identity(x, filters=(256, 1024))
    x = dres_identity(x, filters=(256, 1024))
    x = dres_identity(x, filters=(256, 1024))
    x = dres_identity(x, filters=(256, 1024))
    x = dres_identity(x, filters=(256, 1024))
    x = dres_identity(x, filters=(256, 1024))
    x = dres_identity(x, filters=(256, 1024))


    x = dres_conv(x, s=2, filters=(128, 512))
    x = dres_identity(x, filters=(128, 512))
    x = dres_identity(x, filters=(128, 512))
    x = dres_identity(x, filters=(128, 512))

    x = dres_conv(x, s=1, filters=(64, 256))
    x = dres_identity(x, filters=(64, 256))
    x = dres_identity(x, filters=(64, 256))
    x = Conv2DTranspose(3, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    decoded = Activation(activations.sigmoid)(x)
    return tf.keras.Model(inputs=dec_input, outputs=decoded, name='Decoder')