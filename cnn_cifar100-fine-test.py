from __future__ import print_function
import keras
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.misc
import PIL
import my_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

my_utils.prepare_error_output('fine')

cifar100_label_list = np.array([
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
])

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

batch_size = 128
num_classes = 100
epochs = 12

# input image dimensions
img_rows, img_cols = 32, 32

# Separa os dados de treino e teste para superclasses
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

input_shape = (img_rows, img_cols, 3)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# load weights
model.load_weights("weights-fine.best.hdf5")

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

def get_proper_images(raw):
    raw_float = np.array(raw, dtype=float) 
    images = raw_float.reshape([-1, 3, 32, 32])
    images = images.transpose([0, 2, 3, 1])
    return images

# Executa a validação do score considerando 
# cada classe/superclasse.
for class_idx, _class_ in enumerate(cifar100_label_list):
    x_picked_test = []
    y_picked_test = []

    # Condição para selecionar apenas uma classe/superclasse
    condition = y_test == class_idx
    for idx, cond in enumerate(condition):
        if cond == True:
            x_picked_test.append(x_test[idx])
            y_picked_test.append(y_test[idx])

    x_picked_test = np.array(x_picked_test, dtype=np.float32)
    x_picked_test /= 255

    # converte classe para matrix de binários
    y_picked_test = keras.utils.to_categorical(y_picked_test, num_classes)

    score = model.evaluate(x_picked_test, y_picked_test, verbose=0)

    print("Class: %s | Accuracy: %.2f%%" % (cifar100_label_list[class_idx], score[1]*100))

    y_hat = model.predict(x_picked_test)

    error_count = 0

    for i, index in enumerate(x_picked_test):
        if error_count < 2:
            predict_index = np.argmax(y_hat[i])
            true_index = np.argmax(y_picked_test[i])

            if predict_index != true_index:
                directory = ("./errors/fine/%s" % (cifar100_label_list[true_index]))
                
                if not os.path.exists(directory):
                    os.makedirs(directory)

                rgb = scipy.misc.toimage(x_picked_test[i])
                rgb.save("%s/%si-%i.jpg" % (directory, cifar100_label_list[predict_index], i))

                error_count += 1