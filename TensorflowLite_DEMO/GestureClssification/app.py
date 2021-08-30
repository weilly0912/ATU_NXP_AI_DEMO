
import numpy as np
from tflite_runtime.interpreter import Interpreter 

#load model network
interpreter = Interpreter(model_path='GesturesRecognizer_uint8.tflite')
interpreter.allocate_tensors() 
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
width = input_details[0]['shape'][2]
height = input_details[0]['shape'][1]

#Gesture Clssification
#class_names = ["down", "palm", "l", "fist", "fist_moved", "thumb", "index", "ok", "palm_moved", "c"]
class_names = ['Two', 'Five', 'Yo', 'Blank', 'Fist', 'ThumbsUp']

import cv2
cap = cv2.VideoCapture(1)

# -> back ground method
fgbg  = cv2.bgsegm.createBackgroundSubtractorMOG(history=30, nmixtures=20, backgroundRatio=0.3, noiseSigma=0)
#fgbg  = cv2.createBackgroundSubtractorMOG2()
#fgbg  = cv2.BackgroundSubtractorKNN()

while(True):

  #preprocess image
  ret, frame = cap.read()
  fgmask = fgbg.apply(frame)
  fgmask_resized = cv2.resize(fgmask, (width, height))
  fgmask_resized = cv2.dilate(fgmask_resized, np.ones((3,3), np.uint8), iterations = 2)
  fgmask_resized = cv2.cvtColor(fgmask_resized, cv2.COLOR_GRAY2BGR)
  input_data = np.expand_dims(fgmask_resized, axis=0)#input_data = np.expand_dims(input_data, axis=3)
  interpreter.set_tensor(input_details[0]['index'], input_data) 
  interpreter.invoke()

  #predict
  predict = interpreter.get_tensor(output_details[0]['index'])
  predicted_label = class_names[np.argmax(predict)]
  #print(predicted_label)

  #draw on text
  text = str(predicted_label)
  cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)

  #imshow
  fgmask_resized_show = cv2.resize(fgmask_resized, (frame.shape[1], frame.shape[0]))
  image_show = np.concatenate((frame, fgmask_resized_show)) 
  cv2.imshow('frame', image_show)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()

