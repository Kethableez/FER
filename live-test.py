import cv2
from keras.models import load_model, model_from_json
import os
import cv2
import numpy as np
from keras.models import load_model, model_from_json
from keras.utils import load_img, img_to_array
from keras.preprocessing import image

model = model_from_json(open('./models/model.json', 'r').read())
model.load_weights('./models/model.h5')

face_haar_cascade = cv2.CascadeClassifier('./models/haar.xml')
cap=cv2.VideoCapture(0)

while cap.isOpened():
  res,frame=cap.read()
  height, width , channel = frame.shape
  sub_img = frame[0:int(height/6),0:int(width)]
  black_rect = np.ones(sub_img.shape, dtype=np.uint8)*0
  res = cv2.addWeighted(sub_img, 0.77, black_rect,0.23, 0)
  FONT = cv2.FONT_HERSHEY_SIMPLEX
  FONT_SCALE = 0.8
  FONT_THICKNESS = 2
  lable_color = (10, 10, 255)
  gray_image= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = face_haar_cascade.detectMultiScale(gray_image )
  try:
    for (x,y, w, h) in faces:
      cv2.rectangle(frame, pt1 = (x,y),pt2 = (x+w, y+h), color = (255,0,0),thickness =  2)
      roi_gray = gray_image[y-5:y+h+5,x-5:x+w+5]
      roi_gray=cv2.resize(roi_gray,(48,48))
      image_pixels = img_to_array(roi_gray)
      image_pixels = np.expand_dims(image_pixels, axis = 0)
      image_pixels /= 255
      predictions = model.predict(image_pixels)
      max_index = np.argmax(predictions[0])
      emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
      emotion_prediction = emotion_detection[max_index]
      lable_violation = 'Confidence: {}'.format(str(np.round(np.max(predictions[0])*100,1))+ "%")
      violation_text_dimension = cv2.getTextSize(lable_violation,FONT,FONT_SCALE,FONT_THICKNESS )[0]
      violation_x_axis = int(res.shape[1]- violation_text_dimension[0])
  except :
    pass
  frame[0:int(height/6),0:int(width)] =res
  cv2.imshow('frame', frame)


  if cv2.waitKey(1) & 0xFF == ord('q'):
    break