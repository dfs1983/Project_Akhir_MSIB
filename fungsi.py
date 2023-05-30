import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import RandomRotation, RandomZoom, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

def make_model():
    BATCH_SIZE = 32
    IMG_SIZE = (128, 128)

    data_augmentation = Sequential([
        RandomRotation(factor=(-0.15, 0.15)),
        RandomZoom(height_factor=(-0.3, -0.1))
    ])

    assert(data_augmentation.layers[0].name.startswith('random_rotation'))
    assert(data_augmentation.layers[0].factor == (-0.15, 0.15))
    assert(data_augmentation.layers[1].name.startswith('random_zoom'))
    assert(data_augmentation.layers[1].height_factor == (-0.3, -0.1))

    image_shape = IMG_SIZE + (3,)
    base_model = EfficientNetB0(input_shape=image_shape,
                                include_top=False, 
                                weights='imagenet')

    base_model.trainable = True
    for layer in base_model.layers[:218]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=image_shape)
    x = data_augmentation(inputs)
    x = Rescaling(scale=1./255)(x)  # Rescaling input values
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(units=5, activation="softmax")(x)

    model = Model(inputs, outputs)

    return model
