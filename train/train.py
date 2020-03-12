from datetime import datetime
import os

import keras
from keras.applications.vgg16 import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from lib.tools import save_history, show_image_tile

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = "./data/"
LOG_PATH = os.path.join("log/", datetime.now().strftime('%Y%m%d_%H%M%S'))

IMAGE_SIZE = 96
BATCH_SIZE = 100
NUM_EPOCH = 10


class CustomImageDataGenerator(ImageDataGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def random_transform(self, x, seed=1):
        x = super().random_transform(x, seed=seed)
        # xはnp.float32型になっている
        x = self.addGaussianNoise(x, seed=seed)

        return x

    def addGaussianNoise(self, src, seed):
        row, col, ch = src.shape
        mean = 0
        var = 0.1
        sigma = 15
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = src + gauss

        return noisy


def create_model():
    input_data = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    vgg16_model = VGG16(include_top=False)
    # vgg16_model.trainable = True
    for layer in vgg16_model.layers[:15]:
        layer.trainable = False
    flatten = layers.Flatten()(vgg16_model(input_data))
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
    data_gen = ImageDataGenerator(rescale=1. / 255)
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
