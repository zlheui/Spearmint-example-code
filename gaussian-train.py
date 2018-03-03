import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.datasets import cifar10
from keras.applications.vgg16 import VGG16
from keras import backend as K
from keras import optimizers
from keras import regularizers
import numpy as np
import os.path
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping


def main(job_id, params):
    
    batch_size = int(params['batch_size'][0])
    dropout1 = float(params['dropout'][0])
    dropout2 = float(params['dropout'][1])
    weight_decay = float(params['weight_decay'][0])
    init_std = float(params['init_std'][0])
    lr = float(params['lr'][0])
    momentum = float(params['momentum'][0])
    activation = params['activation'][0]


    img_height, img_width = 32, 32
    epochs = 1000

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_val = x_train[40000:]
    x_train = x_train[0:40000]

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    y_val = y_train[40000:]
    y_train = y_train[0:40000]


    init = keras.initializers.RandomNormal(mean=0.0, stddev=init_std, seed=None)

    model = Sequential()
    # Block 1
    model.add(layers.Conv2D(96, (3, 3), activation=activation, padding='same', name='block1_conv1', kernel_initializer=init, bias_initializer=init, input_shape=(img_height, img_width, 3), kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Conv2D(96, (3, 3), activation=activation, padding='same', name='block1_conv2', kernel_initializer=init, bias_initializer=init, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block1_pool'))
    model.add(Dropout(dropout1))

    # Block 2
    model.add(layers.Conv2D(192, (3, 3), activation=activation, padding='same', name='block2_conv1', kernel_initializer=init, bias_initializer=init, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Conv2D(192, (3, 3), activation=activation, padding='same', name='block2_conv2', kernel_initializer=init, bias_initializer=init, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Conv2D(192, (3, 3), activation=activation, padding='same', name='block2_conv3', kernel_initializer=init, bias_initializer=init, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool'))
    model.add(Dropout(dropout2))


    # Block 3
    model.add(layers.Conv2D(192, (3, 3), activation=activation, padding='same', name='block3_conv1', kernel_initializer=init, bias_initializer=init, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Conv2D(192, (1, 1), activation=activation, padding='same', name='block3_conv2', kernel_initializer=init, bias_initializer=init, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Conv2D(10, (1, 1), activation=activation, padding='same', name='block3_conv3', kernel_initializer=init, bias_initializer=init, kernel_regularizer=regularizers.l2(weight_decay)))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(10, activation='softmax', name='predictions', kernel_initializer=init, bias_initializer=init, kernel_regularizer=regularizers.l2(weight_decay)))


    sgd = optimizers.SGD(lr=lr, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    path_to_weights = "/home/zhulei/spearmint/spearmint/examples/gaussian4-exp-rdshare/weights/"
    # # load current best weight. comment away to share weight
    path_to_best_result = "/home/zhulei/spearmint/spearmint/examples/gaussian4-exp-rdshare/best_job_and_result.txt"

    if os.path.isfile(path_to_best_result):
        with open(path_to_best_result, 'r') as f:
            lines = f.readlines()
            best_job_id = lines[1].strip().split(" ")[1]
            print(best_job_id)
            if int(best_job_id) != -1:
                path_to_best_weights = path_to_weights + "weights_" + best_job_id + ".h5"
                model.load_weights(path_to_best_weights)
                print("loaded weight from " + path_to_best_weights)


    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, verbose=1),
        ModelCheckpoint(path_to_weights+"weights_"+str(job_id)+".h5", monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1),
    ]

    train_history = model.fit(x_train, y_train, batch_size=batch_size, verbose=2, epochs=epochs, validation_data=(x_val, y_val), callbacks=callbacks)
    loss = train_history.history['loss']

    print(loss)
    print(min(loss))

    return min(loss)

if __name__ == "__main__":
    main(23, {'batch_size': ['128'], 'dropout': ['0.3', '0.3'], 'weight_decay': ['0.0005'], 'init_std': ['0.01'], 'lr': ['0.001'], 'momentum': ['0.9'], 'activation': ['relu']})
