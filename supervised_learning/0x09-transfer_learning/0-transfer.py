#!/usr/bin/env python3
"""Transfer Learning"""


import tensorflow as tf
import tensorflow.keras as K
import numpy as np


def preprocess_data(X, Y):
    """function that pre-processes the data"""
    X = K.applications.densenet.preprocess_input(X)
    Y = K.utils.to_categorical(Y)
    return X, Y


if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)
    input_tensor = K.Input(shape=(32, 32, 3))
    resized_images = K.layers.Lambda(
        lambda image: tf.image.resize(image, (224, 224)))(input_tensor)

    base_model = K.applications.DenseNet201(include_top=False,
                                            weights='imagenet',
                                            input_tensor=resized_images,
                                            input_shape=(224, 224, 3),
                                            pooling='max',
                                            classes=1000)
    output = base_model.layers[-1].output
    base_model = K.models.Model(inputs=input_tensor, outputs=output)

    train_datagen = K.preprocessing.image.ImageDataGenerator()
    train_generator = train_datagen.flow(x_train,
                                         y_train,
                                         batch_size=32,
                                         shuffle=False)
    features_train = base_model.predict(train_generator)

    val_datagen = K.preprocessing.image.ImageDataGenerator()
    val_generator = val_datagen.flow(x_test,
                                     y_test,
                                     batch_size=32,
                                     shuffle=False)
    features_valid = base_model.predict(val_generator)

    initializer = K.initializers.he_normal()
    input_tensor = K.Input(shape=features_train.shape[1])

    layer_256 = K.layers.Dense(units=256,
                               activation='elu',
                               kernel_initializer=initializer,
                               kernel_regularizer=K.regularizers.l2())
    output = layer_256(input_tensor)
    dropout = K.layers.Dropout(0.5)
    output = dropout(output)

    softmax = K.layers.Dense(units=10,
                             activation='softmax',
                             kernel_initializer=initializer,
                             kernel_regularizer=K.regularizers.l2())
    output = softmax(output)

    model = K.models.Model(inputs=input_tensor, outputs=output)

    model.compile(optimizer=K.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    lr_reduce = K.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                              factor=0.6,
                                              patience=2,
                                              verbose=1,
                                              mode='max',
                                              min_lr=1e-7)

    early_stop = K.callbacks.EarlyStopping(monitor='val_accuracy',
                                           patience=3,
                                           verbose=1,
                                           mode='max')

    checkpoint = K.callbacks.ModelCheckpoint('cifar10.h5',
                                             monitor='val_accuracy',
                                             verbose=1,
                                             save_weights_only=False,
                                             save_best_only=True,
                                             mode='max',
                                             save_freq='epoch')

    history = model.fit(features_train, y_train,
                        batch_size=32,
                        epochs=20,
                        verbose=1,
                        callbacks=[lr_reduce, early_stop, checkpoint],
                        validation_data=(features_valid, y_test),
                        shuffle=True)
