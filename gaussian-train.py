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

    # # load current best weight
    path_to_weights = "/home/zhulei/spearmint/spearmint/examples/gaussian4-exp-rdshare/weights/"
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


    # datagen = ImageDataGenerator(
    #             featurewise_center=False,  # set input mean to 0 over the dataset
    #             samplewise_center=False,  # set each sample mean to 0
    #             featurewise_std_normalization=False,  # divide inputs by std of the dataset
    #             samplewise_std_normalization=False,  # divide each input by its std
    #             zca_whitening=False)  # apply ZCA whitening
    #             #rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    #             #width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    #             #height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    #             #horizontal_flip=True,  # randomly flip images
    #             #vertical_flip=False)  # randomly flip images
    #         # (std, mean, and principal components if ZCA whitening is applied).

    # datagen.fit(x_train)


    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, verbose=1),
        ModelCheckpoint(path_to_weights+"weights_"+str(job_id)+".h5", monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1),
    ]

    # model.fit(X_train.astype('float32'), Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
    #       shuffle=True, verbose=1, validation_data=(X_valid, Y_valid),
    #       callbacks=callbacks)

    # model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), verbose=2, steps_per_epoch=len(x_train)/batch_size, validation_data=(x_val, y_val), epochs=epochs)


    train_history = model.fit(x_train, y_train, batch_size=batch_size, verbose=2, epochs=epochs, validation_data=(x_val, y_val), callbacks=callbacks)
    loss = train_history.history['loss']

    print(loss)
    print(min(loss))

    return min(loss)



    # # save weights for weight sharing
    # model.save_weights(path_to_weights+"weights_"+str(job_id)+".h5")

    # predicted_x = model.predict(x_test)
    # residuals = np.argmax(predicted_x,1)!=np.argmax(y_test,1)

    # accuracy = np.empty(residuals.shape)

    # for i in range(0, len(residuals)):
    #     if residuals[i]:
    #         accuracy[i] = 1


    # loss = sum(accuracy)/len(residuals)
    # print("the validation 0/1 loss is: ",loss)

    # K.clear_session()

    # return loss


if __name__ == "__main__":
    main(23, {'batch_size': ['128'], 'dropout': ['0.3', '0.3'], 'weight_decay': ['0.0005'], 'init_std': ['0.01'], 'lr': ['0.001'], 'momentum': ['0.9'], 'activation': ['relu']})
