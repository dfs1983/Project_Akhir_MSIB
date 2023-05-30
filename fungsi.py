import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import RandomRotation, RandomZoom
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras import applications
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

def make_model():
    BATCH_SIZE = 32
    IMG_SIZE = (128, 128)
    data_augmentation = Sequential()
    data_augmentation.add(RandomRotation(factor=(-0.15, 0.15)))
    data_augmentation.add(RandomZoom((-0.3, -0.1)))
    
    return data_augmentation
    
    data_augmentation = data_augmentar()
    assert(data_augmentation.layers[0].name.startswith('random_rotation'))
    assert(data_augmentation.layers[0].factor == (-0.15, 0.15))
    assert(data_augmentation.layers[1].name.startswith('random_zoom'))
    assert(data_augmentation.layers[1].height_factor == (-0.3, -0.1))
    
    image_shape=IMG_SIZE
    data_augmentation=data_augmentar()
    IMG_SHAPE = image_shape + (3,)
    base_model = applications.EfficientNetB0(input_shape=IMG_SHAPE,
                                             include_top=False, 
                                             weights='imagenet')
    
    base_model.trainable = True
    for layer in base_model.layers[0:218]:
        layer.trainable = False
    
    inputs = Input(shape=IMG_SHAPE)
    x = data_augmentation(inputs)
    x = preprocess_input(inputs)
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(units=5, activation = "softmax")(x)
    
    model = Model(inputs, outputs)
    
    return model
