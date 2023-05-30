import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import random_rotation, random_zoom
import matplotlib.pyplot as plt

# Menentukan BATCH_SIZE dan IMG_SIZE
BATCH_SIZE = 32
IMG_SIZE = (128, 128)

# Menentukan SEED dan menggunakan set_seed untuk mengatur seed
SEED = 0
tf.random.set_seed(SEED)

# Mendefinisikan directory dataset
directory = '/kaggle/input/wayang-bagong-cepot-gareng-petruk-semar/images/images/train'
val_directory = '/kaggle/input/wayang-bagong-cepot-gareng-petruk-semar/images/images/val'

# Membuat train_dataset menggunakan image_dataset_from_directory
train_dataset = image_dataset_from_directory(directory=directory,
                                             label_mode='categorical',
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             shuffle=True,
                                             validation_split=0.2,
                                             subset='training',
                                             seed=SEED)

# Membuat full_validation_dataset menggunakan image_dataset_from_directory
full_validation_dataset = image_dataset_from_directory(directory=directory,
                                                       label_mode='categorical',
                                                       batch_size=BATCH_SIZE,
                                                       image_size=IMG_SIZE,
                                                       shuffle=True,
                                                       validation_split=0.2,
                                                       subset='validation',
                                                       seed=SEED)

# Menghitung jumlah batch pada full_validation_dataset
validation_batches = len(full_validation_dataset)
print(f'Total number of full_validation_dataset batches : {validation_batches}')

# Membagi full_validation_dataset menjadi validation_dataset dan test_dataset
validation_dataset = full_validation_dataset.take(validation_batches // 2)
test_dataset = full_validation_dataset.skip(validation_batches // 2)

# Menampilkan jumlah batch pada validation_dataset dan test_dataset
print(f'Number of batches in validation dataset : {len(validation_dataset)}')
print(f'Number of batches in test dataset : {len(test_dataset)}')

def data_augmentar():
    """This function applies two data augmentation techniques.
        First, augmentation with RandomRotation.
        Second, augmentation with RandomZoom
    """
    data_augmentation = Sequential([
        RandomRotation(factor=(-0.15, 0.15)),
        RandomZoom(height_factor=(-0.3, -0.1))
    ])
    
    return data_augmentation

data_augmentation = data_augmentar()
assert(data_augmentation.layers[0].name.startswith('random_rotation'))
assert(data_augmentation.layers[0].factor == (-0.15, 0.15))
assert(data_augmentation.layers[1].name.startswith('random_zoom'))
assert(data_augmentation.layers[1].height_factor == (-0.3, -0.1))

# applying data augmentation with a sample image.
plt.figure(figsize=(5, 5))
for images, labels in train_dataset.take(1):
    image = images[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        augmented_image = data_augmentation(image)
        plt.imshow(augmented_image.numpy().astype('uint8'), cmap='gray')
        plt.axis('off')
        plt.show()

def alzheimer_classifier(image_shape=IMG_SIZE, data_augmentation=data_augmentar()):
    """This function creates a classifier for Alzheimer disease MRI images.
    
    Arguments:
        image_shape -> the size of the image in the form (height, width).
        data_augmentation -> the data augmentation object to apply on the training data.
        
    Returns:
        model -> the created classifier.
    """
    IMG_SHAPE = image_shape + (3,)
    inputs = Input(shape=IMG_SHAPE)
    x = data_augmentation(inputs)
    x = x / 255.0  # Normalize pixel values to [0, 1]
    
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=3, input_shape=IMG_SHAPE, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='softmax'))
    
    outputs = model(x)
    
    model = Model(inputs, outputs)
    
    return model
