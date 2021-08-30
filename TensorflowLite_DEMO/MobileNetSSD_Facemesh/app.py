# code src : https://github.com/google-coral/project-posenet
import cv2
import numpy as np
import math
from enum import Enum
from tflite_runtime.interpreter import Interpreter 
'''
facemesh_uint8.tflite->TF2.5
ValueError: Didn't find op for builtin opcode 'MIRROR_PAD' version '2'

'''
#load model network 
interpreterFaceMesh = Interpreter(model_path='facemesh.tflite')
interpreterFaceMesh.allocate_tensors() 
input_details = interpreterFaceMesh.get_input_details()
output_details = interpreterFaceMesh.get_output_details()
width = input_details[0]['shape'][2]
height = input_details[0]['shape'][3]

#API
mappings = open("landmark_contours.txt").readlines()
contours = {}
for line in mappings:
    line = line.strip().split(" ")
    contours[line[0]] = [int(i) for i in line[1:]]


cap = cv2.VideoCapture(1)
while(True):
  
  #ret, frame = cap.read()
  
  #load image and set tensor
  frame         = cv2.imread("face_example.jpg")
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  frame_resized = cv2.resize(frame_rgb, (width, height))#frame_resized = np.array(frame_resized, dtype=np.float32)
  input_data = np.expand_dims(frame_resized.astype("float32"), axis=0)#.astype("float32")
  input_data = input_data.swapaxes(1, 3)
  interpreterFaceMesh.set_tensor(input_details[0]['index'], input_data) 
 
  #start interpreter
  interpreterFaceMesh.invoke()

  #predict
  landmarks_result = interpreterFaceMesh.get_tensor(output_details[0]['index'])
  landmarks_result = np.reshape(landmarks_result, (-1, 468, 3))[:, :, :2] # landmarks_result.shape[1] = 1404 / 3  

  for land in landmarks_result :
        for pt in land:
            cv2.circle(frame_resized, (pt[0], pt[1]), 1, (0, 0, 255), -1)

  cv2.imshow('frame', frame_resized)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()

