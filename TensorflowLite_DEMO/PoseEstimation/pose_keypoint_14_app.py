
import cv2
import numpy as np
import math
from enum import Enum
from tflite_runtime.interpreter import Interpreter 

#load model network 
interpreterPoseEstimation = Interpreter(model_path='pose_detect.tflite')
interpreterPoseEstimation.allocate_tensors() 
input_details = interpreterPoseEstimation.get_input_details()
output_details = interpreterPoseEstimation.get_output_details()
width = input_details[0]['shape'][2]
height = input_details[0]['shape'][1]

#API
class Person:
  def __init__(self):
      self.keyPoints = []
      self.score = 0.0

class Position:
  def __init__(self):
    self.x = 0
    self.y = 0

class BodyPart(Enum):
  TOP = 0,
  NECK = 1,
  RIGHT_SHOULDER = 2,
  RIGHT_ELBOW = 3,
  RIGHT_WRIST = 4,
  LEFT_SHOULDER = 5,
  LEFT_ELBOW = 6,
  LEFT_WRIST = 7,
  RIGHT_HIP = 8,
  RIGHT_KNEE = 9,
  RIGHT_ANKLE = 10,
  LEFT_HIP = 11,
  LEFT_KNEE = 12,
  LEFT_ANKLE = 13,

class KeyPoint:
  def __init__(self):
    self.bodyPart = BodyPart.TOP
    self.position = Position()
    self.score = 0.0

def sigmoid(x):
  return 1. / (1. + math.exp(-x))

cap = cv2.VideoCapture(1)
while(True):

  ret, frame = cap.read()
  
  #load image and set tensor
  #frame         = cv2.imread("pose_test_image.jpg")
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  frame_resized = cv2.resize(frame_rgb, (width, height))
  frame_resized = np.array(frame_resized, dtype=np.float32)
  input_data = np.expand_dims(frame_resized, axis=0)
  interpreterPoseEstimation.set_tensor(input_details[0]['index'], input_data) 
 
  #start interpreter
  interpreterPoseEstimation.invoke()

  #predict
  heat_maps = interpreterPoseEstimation.get_tensor(output_details[0]['index'])

  # output step 1 : heatmap
  height_ = heat_maps.shape[1]
  width_ = heat_maps.shape[2]
  num_key_points = heat_maps.shape[3]
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
    y_coords[i] = (position[0] / float(height_ - 1)*height)
    x_coords[i] = (position[1] / float(width_ - 1 )*width )
    confidence_scores[i] = heat_maps[0][position_y][position_x][i]

  # output step 4 :  Person Score
  person = Person()
  key_point_list = []
  total_score = 0

  for i in range(num_key_points):
    key_point = KeyPoint()
    key_point_list.append(key_point)

  for i, body_part in enumerate(BodyPart):
    key_point_list[i].bodyPart = body_part
    key_point_list[i].position.x = x_coords[i]
    key_point_list[i].position.y = y_coords[i]
    key_point_list[i].score = confidence_scores[i]
    total_score += confidence_scores[i]

  person.keyPoints = key_point_list
  person.score = total_score / num_key_points
  body_joints = [[BodyPart.LEFT_WRIST, BodyPart.LEFT_ELBOW],
                 [BodyPart.LEFT_ELBOW, BodyPart.LEFT_SHOULDER],
                 [BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER],
                 [BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW],
                 [BodyPart.RIGHT_ELBOW, BodyPart.RIGHT_WRIST],
                 [BodyPart.LEFT_SHOULDER, BodyPart.LEFT_HIP],
                 [BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP],
                 [BodyPart.RIGHT_HIP, BodyPart.RIGHT_SHOULDER],
                 [BodyPart.LEFT_HIP, BodyPart.LEFT_KNEE],
                 [BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE],
                 [BodyPart.LEFT_KNEE, BodyPart.LEFT_ANKLE],
                 [BodyPart.RIGHT_KNEE, BodyPart.RIGHT_ANKLE],
                 [BodyPart.TOP, BodyPart.NECK]]

  # draw
  for line in body_joints:
    if person.keyPoints[line[0].value[0]].score > 0.3 and person.keyPoints[line[1].value[0]].score > 0.3:
      start_point_x = int(person.keyPoints[line[0].value[0]].position.x * frame.shape[1]/width  )
      start_point_y = int(person.keyPoints[line[0].value[0]].position.y * frame.shape[0]/height )
      end_point_x   = int(person.keyPoints[line[1].value[0]].position.x * frame.shape[1]/width  )
      end_point_y   = int(person.keyPoints[line[1].value[0]].position.y * frame.shape[0]/height )
      cv2.line(frame, (start_point_x, start_point_y) , (end_point_x, end_point_y), (255, 255, 0), 3)
  
  print("done!"," score=",person.score)

  cv2.imshow('frame', frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()

