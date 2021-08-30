#load model network
import numpy as np
from tflite_runtime.interpreter import Interpreter 
interpreter = Interpreter(model_path='mobilenetssd_uint8.tflite')
interpreter.allocate_tensors() 
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
width = input_details[0]['shape'][2]
height = input_details[0]['shape'][1]

#loading...data and interpreter
import cv2
frame = cv2.imread('image7.jpg')
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_resized = cv2.resize(frame_rgb, (width, height))
input_data = np.expand_dims(frame_resized, axis=0)
interpreter.set_tensor(input_details[0]['index'], input_data) 
interpreter.invoke()

detection_boxes = interpreter.get_tensor(output_details[0]['index'])
detection_classes = interpreter.get_tensor(output_details[1]['index'])
detection_scores = interpreter.get_tensor(output_details[2]['index'])
num_boxes = interpreter.get_tensor(output_details[3]['index'])
for i in range(10):
  if detection_scores[0, i] > .5:
    x = detection_boxes[0, i, [1, 3]] * frame_rgb.shape[1]
    y = detection_boxes[0, i, [0, 2]] * frame_rgb.shape[0]
    rectangle = [x[0], y[0], x[1], y[1]]
    class_id = detection_classes[0, i]
    cv2.rectangle(frame_rgb, (x[0], y[0]), (x[1], y[1]), (0, 255, 0), 2)
    print("box=",rectangle,'class_id=',class_id)

cv2.imshow('TOTORO',frame_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()