import cv2 

PATH = './dataset/valid/sad'
cap = cv2.VideoCapture(0)
iter = 0
while cap.isOpened():
  ret, frame = cap.read()

  frame = frame[120:120 + 250, 200: 200 + 250,:]
  cv2.imshow('Collecting', frame)
  if cv2.waitKey(1) & 0XFF == ord('c'):
    cv2.imwrite(PATH + '/sad-{}.jpg'.format(iter+1), frame)
    iter += 1
    print('{} image was saved to {}'.format(iter, PATH))
    if iter == 80:
      break
  
  if cv2.waitKey(1) & 0Xff == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()