#load model network
import numpy as np
from tflite_runtime.interpreter import Interpreter 
interpreter = Interpreter(model_path='segmentation_uint8.tflite')
interpreter.allocate_tensors() 
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
width = input_details[0]['shape'][2]
height = input_details[0]['shape'][1]


import cv2
cap = cv2.VideoCapture(1)

while(True):
 
  ret, frame = cap.read()
 
  #load image and set tensor
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  frame_resized = cv2.resize(frame_rgb, (width, height))
  input_data = np.expand_dims(frame_resized, axis=0)
  interpreter.set_tensor(input_details[0]['index'], input_data) 

 
  #start interpreter
  interpreter.invoke()

  #predict
  seg = interpreter.get_tensor(output_details[0]['index'])

  cv2.imshow('frame', seg[0])
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()

