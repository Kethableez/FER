import numpy as np
import pandas as pd
import tensorflow as tf
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

DATASET = './dataset/fer2013.csv'
CLASSES = 7
SIZE = 48
BATCH_SIZE = 64

def getConsts():
    return (CLASSES, SIZE, BATCH_SIZE)

def initData():
    df = pd.read_csv('./dataset/fer2013.csv')
    k = np.array(list(map(int,df.iloc[0,1].split(" "))),dtype='uint8').reshape((SIZE,SIZE))

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for index, row in df.iterrows():
        k = row['pixels'].split(" ")
        if row['Usage'] == 'Training':
            X_train.append(np.array(k))
            y_train.append(row['emotion'])
        elif row['Usage'] == 'PublicTest':
            X_test.append(np.array(k))
            y_test.append(row['emotion'])

    X_train = np.array(X_train, dtype = 'uint8')
    y_train = np.array(y_train, dtype = 'uint8')
    X_test = np.array(X_test, dtype = 'uint8')
    y_test = np.array(y_test, dtype = 'uint8')

    X_train = X_train.reshape(X_train.shape[0], SIZE, SIZE, 1)
    X_test = X_test.reshape(X_test.shape[0], SIZE, SIZE, 1)
    y_train= to_categorical(y_train, num_classes=CLASSES)
    y_test = to_categorical(y_test, num_classes=CLASSES)
    return (X_train, y_train), (X_test, y_test)

def initFlow(X_train, y_train, X_test, y_test):
  datagen = ImageDataGenerator( 
      rescale=1./255,
      rotation_range = 10,
      horizontal_flip = True,
      width_shift_range=0.1,
      height_shift_range=0.1,
      fill_mode = 'nearest')


  testgen = ImageDataGenerator( 
      rescale=1./255
      )

  datagen.fit(X_train)

  train_flow = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE) 
  test_flow = testgen.flow(X_test, y_test, batch_size=BATCH_SIZE)
  return (train_flow, test_flow)