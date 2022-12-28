import os

import cv2
import numpy as np
from keras.models import load_model, model_from_json
from keras.preprocessing import image
from keras.utils import img_to_array, load_img

VERSION = 1

EMOTIONS = ('happy', 'neutral', 'sad')

def loadModel():
  model = model_from_json(open('./models/model-v{}.json'.format(VERSION), 'r').read())
  model.load_weights('./models/model-v{}.h5'.format(VERSION))
  return model

def getPath(filename):
  prefix = filename.split('-')[0]
  filename =  filename if filename.endswith('.jpg') else '{}.jpg'.format(filename)
  return '{}/{}'.format(prefix, filename)

def processImg(filename):
  im = cv2.imread('./dataset-v2/test/{}'.format(getPath(filename)))
  im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  im = cv2.resize(im, (100, 100))
  imPixs = img_to_array(im)
  imPixs = np.expand_dims(imPixs, axis = 0)
  imPixs /= 255
  return imPixs

ferModel = loadModel()
while True:
  inpt = input('image name or q \n>> \t')
  if (str(inpt) == 'q'):
    break
  img = processImg(inpt)
  prediction = ferModel.predict(img)[0]
  
  result = {
    'happy': prediction[0],
    'neutral': prediction[1],
    'sad': prediction[2]
  }

  index = np.argmax(prediction)
  predictedEmotion = EMOTIONS[index]
  print('[Mood: {0: <10}] [Conf: {1} %]'.format(predictedEmotion, str(np.round(np.max(prediction)*100,1))))
  print()

