import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from const import Const

from ferModel import ferModelV2
from modelFlow import getConsts, initData, initFlow

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

TRAIN_DIR = os.path.join('dataset-v2', 'train')
TEST_DIR = os.path.join('dataset-v2', 'test')
VALID_DIR = os.path.join('dataset-v2', 'valid')

trainDatagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=.1, validation_split=0.2)
testDatagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
trainingSet = trainDatagen.flow_from_directory(TRAIN_DIR, batch_size=28, target_size=(100, 100), shuffle=True, color_mode='grayscale', class_mode='categorical', subset='training')
validationSet = trainDatagen.flow_from_directory(VALID_DIR, batch_size=28, target_size=(100, 100), shuffle=True, color_mode='grayscale', class_mode='categorical', subset='validation')
testSet = testDatagen.flow_from_directory(TEST_DIR, batch_size=28, target_size=(100, 100), shuffle=True, color_mode='grayscale', class_mode='categorical')

model = ferModelV2()

checkpointer = [
  # EarlyStopping(monitor = 'val_accuracy', verbose = 1, restore_best_weights=True,mode="max", patience = 30),
                ModelCheckpoint(filepath='model.best.hdf5', monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")]

stepsEpoch = trainingSet.n // trainingSet.batch_size
validationSteps = validationSet.n // validationSet.batch_size

history = model.fit(x=trainingSet, validation_data=validationSet, epochs=400, callbacks=[checkpointer], steps_per_epoch=stepsEpoch, validation_steps=validationSteps)

trainLoss = history.history['loss']
valLoss = history.history['val_loss']
trainAcc = history.history['accuracy']
valAcc = history.history['val_accuracy']

epochs = range(len(trainAcc))

plt.plot(epochs,trainLoss,'r', label='Train loss')
plt.plot(epochs,valLoss,'b', label='Val loss')
plt.title('train_loss vs val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.figure()

plt.plot(epochs,trainAcc,'r', label='train_acc')
plt.plot(epochs,valAcc,'b', label='val_acc')
plt.title('train_acc vs val_acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.figure()
plt.show()

model.save('./models/Fer2013.h5')
loss = model.evaluate(testSet) 
print("Test Loss " + str(loss[0]))
print("Test Acc: " + str(loss[1]))

# def plot_confusion_matrix(y_test, y_pred, CLASSES,
#                           normalize=False,
#                           title='Unnormalized confusion matrix',
#                           cmap=plt.cm.Blues):
#   cm = confusion_matrix(y_test, y_pred)
    
#   if normalize:
#     cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)
        
#   np.set_printoptions(precision=2)
      
#   plt.imshow(cm, interpolation='nearest', cmap=cmap)
#   plt.title(title)
#   plt.colorbar()
#   tick_marks = np.arange(len(CLASSES))
#   plt.xticks(tick_marks, CLASSES, rotation=45)
#   plt.yticks(tick_marks, CLASSES)

#   thresh = cm.min() + (cm.max() - cm.min()) / 2.
#   for i in range (cm.shape[0]):
#     for j in range (cm.shape[1]):
#       plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

#   plt.tight_layout()
#   plt.ylabel('True expression')
#   plt.xlabel('Predicted expression')
#   plt.show()

# y_pred_ = model.predict(testSet/255., verbose=1)
# y_pred = np.argmax(y_pred_, axis=1)
# t_te = np.argmax(y_test, axis=1)
# fig = plot_confusion_matrix(y_test=t_te, y_pred=y_pred,
#                       classes=CLASSES,
#                       normalize=True,
#                       cmap=plt.cm.Greys,   title='Average accuracy: ' + str(np.sum(y_pred == t_te)/len(t_te)) + '\n')

version =1
model_json = model.to_json()
with open("models/model-v{}.json".format(version), "w") as json_file:
  json_file.write(model_json)

model.save_weights("models/model-v{}.h5".format(version))
print("Saved model to disk")