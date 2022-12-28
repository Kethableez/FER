import os

import cv2
import numpy as np
from keras.models import load_model, model_from_json
from keras.preprocessing import image
from keras.utils import img_to_array, load_img

VERSION = 1

def loadModel():
  model = model_from_json(open('./models/model-v{}.json'.format(VERSION), 'r').read())
  model.load_weights('./models/model-v{}.h5'.format(VERSION))
  return model


face_haar_cascade = cv2.CascadeClassifier('./models/haar.xml')
cap=cv2.VideoCapture(0)

COLORS = [
  (0, 0, 255),
  (0, 255, 255),
  (255, 0, 255),
  (170, 255, 0),
  (0, 170, 255),
  (170, 0, 255),
  (255, 170, 0),
  (0, 255, 170),
  (255, 0, 170),
  (255, 0, 0,),
  (0, 255, 0),
  (255, 255, 0),
  (0,0,0),
  (255,255,255)
]

ferModel = loadModel()

while cap.isOpened():
  res,frame=cap.read()
  height, width , channel = frame.shape
  sub_img = frame[0:int(height/6),0:int(width)]
  black_rect = np.ones(sub_img.shape, dtype=np.uint8)*0
  res = cv2.addWeighted(sub_img, 0.77, black_rect,0.23, 0)
  FONT = cv2.FONT_HERSHEY_SIMPLEX
  FONT_SCALE = 0.5
  FONT_THICKNESS = 1
  label_color = (10, 10, 255)
  label = "Emotion detection"
  label_dims = cv2.getTextSize(label, FONT, 0.8, FONT_THICKNESS * 2)[0]
  textX = int((res.shape[1] - label_dims[0]) / 2)
  textY = int((res.shape[0] - label_dims[1]) / 2)

  gray_image= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = face_haar_cascade.detectMultiScale(gray_image )
  try:
    iter = 0
    for (x,y, w, h) in faces:
      cv2.rectangle(frame, pt1 = (x,y),pt2 = (x+w, y+h), color = COLORS[len(faces)-iter],thickness =  2)
      roi_gray = gray_image[y-5:y+h+5,x-5:x+w+5]
      roi_gray=cv2.resize(roi_gray,(48,48))
      image_pixels = img_to_array(roi_gray)
      image_pixels = np.expand_dims(image_pixels, axis = 0)
      image_pixels /= 255
      predictions = model.predict(image_pixels)
      max_index = np.argmax(predictions[0])
      emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
      emotion_prediction = emotion_detection[max_index]
      cv2.putText(res, '[Mood: {0: <10}] [Conf: {1} %]'.format(emotion_prediction, str(np.round(np.max(predictions[0])*100,1))), (0, textY + (len(faces)-iter * -18)), FONT, 0.5, COLORS[len(faces)-iter], 1)
      iter += 1

  except :
    pass
  frame[0:int(height/6),0:int(width)] =res
  cv2.imshow('frame', frame)


  if cv2.waitKey(1) & 0xFF == ord('q'):
    break