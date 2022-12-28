from keras.layers import BatchNormalization, Dense, Dropout, InputLayer, Flatten, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from const import Const


def ferModelV2():
  model = Sequential()
  # M1
  model.add(InputLayer((100, 100, 1)))
  model.add(Conv2D(4 * 64, kernel_size=(3,3)))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(4 * 64, kernel_size=(3,3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

  # M2
  model.add(Conv2D(2 * 64, kernel_size=(3,3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(2 * 64, kernel_size=(3,3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

  # M3
  model.add(Conv2D(64, kernel_size=(3,3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(64, kernel_size=(3,3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

  model.add(Flatten())

  # D1
  model.add(Dense(2*2*2*64))
  model.add(BatchNormalization())
  model.add(Activation('relu'))

  # D2
  model.add(Dense(2*2*64))
  model.add(BatchNormalization())
  model.add(Activation('relu'))

  # D3
  model.add(Dense(64))
  model.add(BatchNormalization())
  model.add(Activation('relu'))

  model.add(Dense(3, activation='softmax'))

  model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7), metrics=['accuracy'])
  model.summary()
  plot_model(model, to_file='./graphs/model-v1.png', show_shapes=True, show_layer_names=True)
  return model

# def ferModelV2(input_shape=(48, 48, 1)):
#   visible = Input(shape=input_shape, name='input')
#   num_classes = 7

#   conv1_1 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name = 'conv1_1')(visible)
#   conv1_1 = BatchNormalization()(conv1_1)
#   conv1_2 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name = 'conv1_2')(conv1_1)
#   conv1_2 = BatchNormalization()(conv1_2)
#   pool1_1 = MaxPooling2D(pool_size=(2,2), name = 'pool1_1')(conv1_2)
#   drop1_1 = Dropout(0.3, name = 'drop1_1')(pool1_1)

#   conv2_1 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_1')(drop1_1)
#   conv2_1 = BatchNormalization()(conv2_1)
#   conv2_2 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_2')(conv2_1)
#   conv2_2 = BatchNormalization()(conv2_2)
#   conv2_3 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_3')(conv2_2)
#   conv2_2 = BatchNormalization()(conv2_3)
#   pool2_1 = MaxPooling2D(pool_size=(2,2), name = 'pool2_1')(conv2_3)
#   drop2_1 = Dropout(0.3, name = 'drop2_1')(pool2_1)

#   conv3_1 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_1')(drop2_1)
#   conv3_1 = BatchNormalization()(conv3_1)
#   conv3_2 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_2')(conv3_1)
#   conv3_2 = BatchNormalization()(conv3_2)
#   conv3_3 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_3')(conv3_2)
#   conv3_3 = BatchNormalization()(conv3_3)
#   conv3_4 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_4')(conv3_3)
#   conv3_4 = BatchNormalization()(conv3_4)
#   pool3_1 = MaxPooling2D(pool_size=(2,2), name = 'pool3_1')(conv3_4)
#   drop3_1 = Dropout(0.3, name = 'drop3_1')(pool3_1)

#   #the 4-th block
#   conv4_1 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_1')(drop3_1)
#   conv4_1 = BatchNormalization()(conv4_1)
#   conv4_2 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_2')(conv4_1)
#   conv4_2 = BatchNormalization()(conv4_2)
#   conv4_3 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_3')(conv4_2)
#   conv4_3 = BatchNormalization()(conv4_3)
#   conv4_4 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_4')(conv4_3)
#   conv4_4 = BatchNormalization()(conv4_4)
#   pool4_1 = MaxPooling2D(pool_size=(2,2), name = 'pool4_1')(conv4_4)
#   drop4_1 = Dropout(0.3, name = 'drop4_1')(pool4_1)
  
#   #the 5-th block
#   conv5_1 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_1')(drop4_1)
#   conv5_1 = BatchNormalization()(conv5_1)
#   conv5_2 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_2')(conv5_1)
#   conv5_2 = BatchNormalization()(conv5_2)
#   conv5_3 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_3')(conv5_2)
#   conv5_3 = BatchNormalization()(conv5_3)
#   conv5_4 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_4')(conv5_3)
#   conv5_3 = BatchNormalization()(conv5_3)
#   pool5_1 = MaxPooling2D(pool_size=(2,2), name = 'pool5_1')(conv5_4)
#   drop5_1 = Dropout(0.3, name = 'drop5_1')(pool5_1)

#   flatten = Flatten(name = 'flatten')(drop5_1)
#   ouput = Dense(num_classes, activation='softmax', name = 'output')(flatten)

#   model = Model(inputs =visible, outputs = ouput)
#   print(model.summary())
    
#   return model


# def ferModelV2():
#   model = Sequential()
#   # M1

#   model.add(InputLayer((150, 150, 1)))
#   model.add(Conv2D(64, kernel_size=3, padding='same'))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(Conv2D(64, kernel_size=3, padding='same'))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(MaxPooling2D(pool_size=(2,2)))
#   model.add(Dropout(0.3))

#   # M2
#   model.add(Conv2D(128, kernel_size=3, padding='same'))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(Conv2D(128, kernel_size=3, padding='same'))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(Conv2D(128, kernel_size=3, padding='same'))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(MaxPooling2D(pool_size=(2,2)))
#   model.add(Dropout(0.3))

#   # M3
#   model.add(Conv2D(256, kernel_size=3, padding='same'))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(Conv2D(256, kernel_size=3, padding='same'))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(Conv2D(256, kernel_size=3, padding='same'))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(Conv2D(256, kernel_size=3, padding='same'))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(MaxPooling2D(pool_size=(2,2)))
#   model.add(Dropout(0.3))

#   # M4
#   model.add(Conv2D(256, kernel_size=3, padding='same'))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(Conv2D(256, kernel_size=3, padding='same'))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(Conv2D(256, kernel_size=3, padding='same'))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(Conv2D(256, kernel_size=3, padding='same'))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(MaxPooling2D(pool_size=(2,2)))
#   model.add(Dropout(0.3))

#   # M5
#   model.add(Conv2D(512, kernel_size=3, padding='same'))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(Conv2D(512, kernel_size=3, padding='same'))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(Conv2D(512, kernel_size=3, padding='same'))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(Conv2D(512, kernel_size=3, padding='same'))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(MaxPooling2D(pool_size=(2,2)))
#   model.add(Dropout(0.3))

#   # M3
#   model.add(Conv2D(64, kernel_size=(3,3), input_shape=(250, 250, 1)))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(Conv2D(64, kernel_size=(3,3), padding='same'))
#   model.add(BatchNormalization())
#   model.add(Activation('relu'))
#   model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

#   model.add(Flatten())
#   model.add(Dense(3, activation='softmax'))

#   model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7), metrics=['accuracy'])
#   model.summary()
#   plot_model(model, to_file='./graphs/model-v1.png', show_shapes=True, show_layer_names=True)
#   return model