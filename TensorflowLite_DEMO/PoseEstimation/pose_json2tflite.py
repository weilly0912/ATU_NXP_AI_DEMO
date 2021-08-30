# 已確認可以使用!! 但還未調整輸出結果
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter 

import math
def sigmoid(x):
  return 1. / (1. + math.exp(-x))


#--------------------------------------------------------------------------------------
#load model network 
interpreterPoseEstimation = Interpreter(model_path='posenet.tflite')
interpreterPoseEstimation.resize_tensor_input(interpreterPoseEstimation.get_input_details()[0]['index'], (1, 224, 224, 3))
interpreterPoseEstimation.allocate_tensors() 
input_details = interpreterPoseEstimation.get_input_details()
output_details = interpreterPoseEstimation.get_output_details()
width = input_details[0]['shape'][2]
height = input_details[0]['shape'][1]

#load image and set tensor
frame         = cv2.imread("pose_test_image.jpg")
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_resized = cv2.resize(frame_rgb, (width, height))
frame_resized = np.array(frame_resized, dtype=np.float32)
input_data = np.expand_dims(frame_resized, axis=0)
interpreterPoseEstimation.set_tensor(input_details[0]['index'], input_data) 

#interpreter
interpreterPoseEstimation.invoke()
heat_maps   = interpreterPoseEstimation.get_tensor(output_details[0]['index'])

#--------------------------------------------------------------------------------------
#predict
heat_maps   = interpreterPoseEstimation.get_tensor(output_details[1]['index'])
offset_maps = interpreterPoseEstimation.get_tensor(output_details[0]['index'])
height_ = heat_maps.shape[1]
width_  = heat_maps.shape[2]
num_key_points      = heat_maps.shape[3]
key_point_positions = [[0] * 2 for i in range(num_key_points)]

# output step 2 : key_point_positions
for key_point in range(num_key_points):
    max_val = heat_maps[0][0][0][key_point]
    max_row = 0
    max_col = 0
    for row in range(height_):
      for col in range(width_):
        heat_maps[0][row][col][key_point] = sigmoid(heat_maps[0][row][col][key_point])
        if heat_maps[0][row][col][key_point] > max_val:
          max_val = heat_maps[0][row][col][key_point]
          max_row = row
          max_col = col
        key_point_positions[key_point] = [max_row, max_col]

x_coords = [0] * num_key_points
y_coords = [0] * num_key_points
confidence_scores = [0] * num_key_points

# output step 3 : confidence_scores and x/y_coords
for i, position in enumerate(key_point_positions):
    position_y = int(key_point_positions[i][0])
    position_x = int(key_point_positions[i][1])
    y_coords[i] = (position[0] / float(height_ - 1)*height )
    x_coords[i] = (position[1] / float(width_ - 1 )*width )
    confidence_scores[i] = heat_maps[0][position_y][position_x][i]
    dx = int(position_x * frame.shape[1]/width_ )
    dy = int(position_y * frame.shape[0]/height_ ) 
    cv2.circle(frame, (dx, dy), 1, (0, 0, 255), 10)


while(True):
  cv2.imshow('frame', frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
cv2.destroyAllWindows()