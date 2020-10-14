from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.compat.v1 import ConfigProto, Session
import tensorflow as tf

from os import getcwd

config = ConfigProto()
config.gpu_options.allow_growth = True
session = Session(config=config)

BASE_DIR = getcwd()
# %%

datagen = ImageDataGenerator(rescale=1. / 255, dtype=tf.dtypes.float16)
train_generator = datagen.flow_from_directory(BASE_DIR + '\\chest_xray\\train\\', target_size=(227, 227), batch_size=16,
                                              color_mode='grayscale')
test_generator = datagen.flow_from_directory(BASE_DIR + '\\chest_xray\\test\\', target_size=(227, 227), batch_size=16,
                                             color_mode='grayscale')
val_generator = datagen.flow_from_directory(BASE_DIR + '\\chest_xray\\val\\', target_size=(227, 227), batch_size=16,
                                            color_mode='grayscale')


# %%

# AlexNet
def AlexNet(weights=None) -> Sequential:
    '''
    keras implementation of AlexNet
    :param weights: path to preloaded weights. If none starts with random weights
    :return:
    '''
    model = Sequential()

    # 1ST layer
    model.add(Conv2D(filters=96, strides=4, input_shape=(227, 227, 1), kernel_size=11, activation='relu'))
    model.add(MaxPooling2D(pool_size=3, strides=2))

    # 2nd layer
    model.add(Conv2D(filters=256, padding='same', kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=3, strides=2))

    # 3rd layer
    model.add(Conv2D(filters=384, padding='same', kernel_size=5, activation='relu'))

    # 4th layer
    model.add(Conv2D(filters=384, padding='same', kernel_size=3, activation='relu'))

    # 5th layer
    model.add(Conv2D(filters=256, padding='same', kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=3, strides=2))

    model.add(Flatten())

    # FC Layers
    model.add(Dense(9216, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu'))

    # output layer
    model.add(Dense(2, activation='sigmoid'))

    if weights is not None:
        model.load_weights(weights)

    return model

# %%

# training the Alex Net
model = AlexNet()

optimizer = SGD(lr=.01)
METRICS = [
    'accuracy',
    Precision(name='precision'),
    Recall(name='recall')
]

log_dir = "logs/fit/"
callbacks = [
    EarlyStopping(monitor='loss', patience=2),
    # TensorBoard(log_dir=log_dir, histogram_freq=1)
]

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=METRICS)
model.summary()

model.fit(
    train_generator,
    epochs=30,
    steps_per_epoch=320,
    validation_data=val_generator,
    callbacks=callbacks
)

model.evaluate(test_generator)

model.save(BASE_DIR + '/model')
