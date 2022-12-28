import cv2
import numpy as np
from PIL import Image
from keras.models import model_from_json
from keras.utils import img_to_array

EMOTIONS = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

model = model_from_json(open('./models/model.json', 'r').read())
model.load_weights('./models/model.h5')
face_haar_cascade = cv2.CascadeClassifier('./models/haar.xml')

def detectMood(path):
  try:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x,y,w,h = face_haar_cascade.detectMultiScale(img)
    imgGray=cv2.resize(img[y-5:y+h+5,x-5:x+w+5],(48,48))
    cv2.imshow('im', imgGray)
    cv2.waitKey(0)
    pxls = img_to_array(imgGray)
    pxls = np.expand_dims(pxls, axis=0)
    pxls /= 255
    prediction = model.predict(pxls)[0]
    emotion = EMOTIONS[np.argmax(prediction)]
    confidence = str(np.round(np.max(prediction) * 100, 1))
    print(emotion, confidence)

  except Exception:
    raise Exception('[FFR - MDT] No face were recognised')

for img in ['./test/test-1.png', './test/test-2.png', './test/test-3.png', './test/test-4.jpg', './test/test-5.jpg', './test/test-7.jpeg']:
  detectMood(img)