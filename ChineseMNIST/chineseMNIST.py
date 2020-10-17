'''
This is from a kaggle competition
https://www.kaggle.com/gpreda/chinese-mnist
'''

import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

DATA_DIR = 'D:/kaggle/chineseMNIST/'

IMAGE_SIZE = (64, 64)
train_df = pd.read_csv(DATA_DIR + 'chinese_mnist.csv')

# Prepping the Data
train_df['file'] = train_df.apply(lambda x: f'input_{x[0]}_{x[1]}_{x[2]}.jpg', axis=1)

train_df, test_df = train_test_split(train_df,
                                     test_size=0.2,
                                     stratify=train_df['character'].values)

train_df, val_df = train_test_split(train_df,
                                    test_size=0.1,
                                    stratify=train_df['character'].values)

train_generator = ImageDataGenerator(rescale=1./255, rotation_range=20, color_mode='grayscale')
test_generator = ImageDataGenerator(rescale=1./255)

#fix this up next time im using this
train_data = train_generator.flow_from_dataframe(x_col='file', y_col='value')
val_data = test_generator.flow_from_dataframe(x_col='file', y_col='value')
test_data = test_generator.flow_from_dataframe(x_col='file', y_col='value')

