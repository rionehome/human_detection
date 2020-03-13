from datetime import datetime
import os

import keras
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from lib.tools import save_history

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = "./data/"
LOG_PATH = os.path.join("log/", datetime.now().strftime('%Y%m%d_%H%M%S'))

IMAGE_SIZE = 96
BATCH_SIZE = 100
NUM_EPOCH = 10


class CustomImageDataGenerator(ImageDataGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def flow_from_directory(self,
                            directory,
                            target_size=(256, 256),
                            color_mode='rgb',
                            classes=None,
                            class_mode='categorical',
                            batch_size=32,
                            shuffle=True,
                            seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            subset=None,
                            interpolation='nearest'):
        # 親クラスのflow_from_directory
        batches = super().flow_from_directory(directory, target_size, color_mode, classes, class_mode, batch_size,
                                              shuffle, seed, save_to_dir, save_prefix, save_format, follow_links,
                                              subset, interpolation)
        # 拡張処理
        while True:
            batch_x, batch_y = next(batches)
            batch_x = (0.299 * batch_x[:, :, :, 0] + 0.587 * batch_x[:, :, :, 1] + 0.114 * batch_x[:, :, :, 2]) / 255.
            batch_x = batch_x[:, :, :, np.newaxis]
            # 返り値
            yield batch_x, batch_y


def create_model():
    input_data = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    conv1 = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_data)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)
    conv1 = layers.MaxPool2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation('relu')(conv2)
    conv2 = layers.MaxPool2D(pool_size=(2, 2))(conv2)
    conv3 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation('relu')(conv3)
    conv3 = layers.MaxPool2D(pool_size=(2, 2))(conv3)
    # vgg16_model = VGG16(include_top=False)
    # vgg16_model.trainable = True
    # for layer in vgg16_model.layers[:15]:
    #    layer.trainable = False
    # flatten = layers.Flatten()(vgg16_model(input_data))
    flatten = layers.Flatten()(conv3)
    dense1 = layers.Dense(256, activation='relu')(flatten)
    dense1 = layers.Dropout(0.3)(dense1)
    predict = layers.Dense(2, activation='softmax')(dense1)
    model = models.Model(inputs=input_data, outputs=predict)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.RMSprop(lr=1e-4),
        metrics=['acc']
    )

    return model


def train():
    data_gen = CustomImageDataGenerator()
    train_data_iterator = data_gen.flow_from_directory(
        os.path.join(DATASET_PATH, "train_image"),
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )
    val_data_iterator = data_gen.flow_from_directory(
        os.path.join(DATASET_PATH, "test_image"),
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )
    test_data_iterator = data_gen.flow_from_directory(
        os.path.join(DATASET_PATH, "test_image"),
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    model = create_model()
    model.summary()

    callbacks_list = [
        # keras.callbacks.EarlyStopping(
        #    monitor='val_loss',
        #    patience=10
        # ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(LOG_PATH, "model.h5"),
            monitor="val_acc",
            save_best_only=True
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(LOG_PATH, "tensor_board_log"),
            histogram_freq=0
        )
    ]

    history = model.fit_generator(
        train_data_iterator,
        steps_per_epoch=50,
        epochs=NUM_EPOCH,
        validation_data=val_data_iterator,
        validation_steps=50,
        callbacks=callbacks_list
    )

    save_history(history, LOG_PATH)

    model = models.load_model(os.path.join(LOG_PATH, "model.h5"))

    print(model.evaluate_generator(test_data_iterator))
    print(model.predict_generator(test_data_iterator))


if __name__ == '__main__':
    train()
