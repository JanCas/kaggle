# %%
'''
This is the MNIST Digit dataset
https://www.kaggle.com/c/digit-recognizer
'''


import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, BatchNormalization, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from math import floor

batch_size = 128

train_df = pd.read_csv('digit-recognizer (1)/train.csv')
test_df = pd.read_csv('digit-recognizer (1)/test.csv')

y_train = train_df['label']
x_train = train_df.drop('label', axis=1)
y_train = to_categorical(y_train, num_classes=10)

# reshaping intro matrices
x_train = x_train.values.reshape((-1, 28, 28, 1))
test_df = test_df.values.reshape((-1, 28, 28, 1))

# splitting data into train/val
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.1, random_state=42)

# %%

# creating the generators
train_gen = ImageDataGenerator(rescale=1. / 255, rotation_range=20)
train_gen.fit(x_train)

val_gen = ImageDataGenerator(rescale=1. / 255)
test_gen = ImageDataGenerator(rescale=1. / 255)

train_data = train_gen.flow(x=x_train, y=y_train, batch_size=batch_size)
val_data = val_gen.flow(x_val, y_val, batch_size=batch_size)
test_val_data = test_gen.flow(test_df)


# %%

# Lenet5

def LeNet5(input: Input) -> Sequential:
    """
    LeNet-5 with 1 extra convolutional layer and some Normalization to prevent overfitting
    as well as more filters compared to the normal one
    :param Input:
    :return:
    """
    model = Sequential()
    model.add(input)

    # layer 1
    model.add(Conv2D(filters=32, strides=1, kernel_size=5, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=3, strides=2))

    # layer 2
    model.add(Conv2D(filters=64, strides=1, kernel_size=5, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=128, strides=1, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())

    model.add(Flatten())
    # fully connected layers
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(.4))
    model.add(Dense(84, activation='relu'))

    model.add(Dense(10, activation='softmax'))

    return model


# %%

input = Input(shape=(28, 28, 1))

lenet = LeNet5(input)

opt = RMSprop()
METRICS = [
    'accuracy',
    Precision(name='precision'),
    Recall(name='recall')
]

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=.1, min_lr=.000000001)
]

lenet.compile(loss='categorical_crossentropy', optimizer=opt, metrics=METRICS)
lenet.summary()
history = lenet.fit(
    train_data,
    epochs=50,
    steps_per_epoch=floor(train_data.n / train_data.batch_size),
    validation_data=val_data,
    callbacks=callbacks
)

# %%

# evaluating the model

fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

plt.show()
