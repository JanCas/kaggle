
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
#%%

datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory('chest_xray/train/', target_size=(256, 256), batch_size=64)
test_generator = datagen.flow_from_directory('chest_xray/test/', target_size=(256, 256), batch_size=64)
val_generator = datagen.flow_from_directory('chest_xray/val/', target_size=(256, 256), batch_size=16)

#%%

#AlexNet
def AlexNet():
    model = Sequential()

    #1ST layer
    model.add(Conv2D(filters=96, strides=4, input_shape=(256,256,3), kernel_size=11, activation='relu'))
    model.add(MaxPooling2D(pool_size=3, strides=2))

    #2nd layer
    model.add(Conv2D(filters=256, padding='same', kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=3, strides=2))

    #3rd layer
    model.add(Conv2D(filters=384, padding='same', kernel_size=5, activation='relu'))

    #4th layer
    model.add(Conv2D(filters=384, padding='same', kernel_size=3, activation='relu'))

    #5th layer
    model.add(Conv2D(filters=256, padding='same', kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=3, strides=2))

    model.add(Flatten())

    model.add(Dense(9216, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))

    model.add(Dense(2, activation='sigmoid'))
    #FC Layers

    return model

#%%

#training the model
model = AlexNet()

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), metrics=['accuracy'])
model.summary()

model.fit(
    train_generator,
    epochs=4,
    steps_per_epoch=81,
    validation_steps=1,
    validation_data=val_generator,
)

model.evaluate(test_generator)
