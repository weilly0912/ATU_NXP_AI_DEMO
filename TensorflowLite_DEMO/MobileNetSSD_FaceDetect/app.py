#load model network
import numpy as np
from tflite_runtime.interpreter import Interpreter 
interpreter = Interpreter(model_path='mobilenetssd_uint8_face.tflite')
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
  detection_boxes = interpreter.get_tensor(output_details[0]['index'])
  detection_classes = interpreter.get_tensor(output_details[1]['index'])
  detection_scores = interpreter.get_tensor(output_details[2]['index'])
  num_boxes = interpreter.get_tensor(output_details[3]['index'])
  for i in range(10):
    if detection_scores[0, i] > .5:
      x = detection_boxes[0, i, [1, 3]] * frame_rgb.shape[1]
      y = detection_boxes[0, i, [0, 2]] * frame_rgb.shape[0]
      rectangle = [x[0], y[0], x[1], y[1]]
      cv2.rectangle(frame, (x[0], y[0]), (x[1], y[1]), (0, 255, 0), 2)
      #class_id = detection_classes[0, i]
      #print("box=",rectangle,'class_id=',class_id)


  cv2.imshow('frame', frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()

