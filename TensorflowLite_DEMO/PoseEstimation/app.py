# code src : https://github.com/google-coral/project-posenet
import cv2
import numpy as np
import math
from enum import Enum
from tflite_runtime.interpreter import Interpreter 

#load model network 
interpreterPoseEstimation = Interpreter(model_path='posenet_mobilenet_v1_075_481_641_quant.tflite')
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
    NOSE = 0,
    LEFT_EYE = 1,
    RIGHT_EYE = 2,
    LEFT_EAR = 3,
    RIGHT_EAR = 4,
    LEFT_SHOULDER = 5,
    RIGHT_SHOULDER = 6,
    LEFT_ELBOW = 7,
    RIGHT_ELBOW = 8,
    LEFT_WRIST = 9,
    RIGHT_WRIST = 10,
    LEFT_HIP = 11,
    RIGHT_HIP = 12,
    LEFT_KNEE = 13,
    RIGHT_KNEE = 14,
    LEFT_ANKLE = 15,
    RIGHT_ANKLE = 16,

class KeyPoint:
  def __init__(self):
    self.bodyPart = BodyPart.NOSE
    self.position = Position()
    self.score = 0.0

cap = cv2.VideoCapture(1)
while(True):
  
  ret, frame = cap.read()
  
  #load image and set tensor
  #frame         = cv2.imread("pose_test_image.jpg")
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  frame_resized = cv2.resize(frame_rgb, (width, height))#frame_resized = np.array(frame_resized, dtype=np.float32)
  input_data = np.expand_dims(frame_resized, axis=0)
  interpreterPoseEstimation.set_tensor(input_details[0]['index'], input_data) 
 
  #start interpreter
  interpreterPoseEstimation.invoke()

  #predict
  heat_maps   = interpreterPoseEstimation.get_tensor(output_details[0]['index'])
  offset_maps = interpreterPoseEstimation.get_tensor(output_details[1]['index'])
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
        if heat_maps[0][row][col][key_point] > max_val:
          max_val = heat_maps[0][row][col][key_point]
          max_row = row
          max_col = col
        key_point_positions[key_point] = [max_row, max_col]

  # output step 3 : confidence_scores and x/y_coords
  x_coords = [0] * num_key_points
  y_coords = [0] * num_key_points
  confidence_scores = [0] * num_key_points

  for i, position in enumerate(key_point_positions):
    position_y = int(key_point_positions[i][0])
    position_x = int(key_point_positions[i][1])
    y_coords[i] = int(position[0])
    x_coords[i] = int(position[1])
    confidence_scores[i] = (float)(heat_maps[0][position_y][position_x][i] /255)
    #dx = int(position_x * frame.shape[1]/width_ ). dy = int(position_y * frame.shape[0]/height_ ) ,cv2.circle(frame, (dx, dy), 1, (0, 0, 255), 10)

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
                 [BodyPart.RIGHT_KNEE,BodyPart.RIGHT_ANKLE]
                 ]
  

  # draw body
  for line in body_joints:
    if person.keyPoints[line[0].value[0]].score > 0.4 and person.keyPoints[line[1].value[0]].score > 0.4:
      start_point_x = (int)(person.keyPoints[line[0].value[0]].position.x  * frame.shape[1]/width_)
      start_point_y = (int)(person.keyPoints[line[0].value[0]].position.y  * frame.shape[0]/height_ )
      end_point_x   = (int)(person.keyPoints[line[1].value[0]].position.x  * frame.shape[1]/width_)
      end_point_y   = (int)(person.keyPoints[line[1].value[0]].position.y  * frame.shape[0]/height_ )
      cv2.line(frame, (start_point_x, start_point_y) , (end_point_x, end_point_y), (255, 255, 0), 3)

  # draw head
  left_ear_x   = (int)(person.keyPoints[3].position.x  * frame.shape[1]/width_)
  left_ear_y   = (int)(person.keyPoints[3].position.y  * frame.shape[0]/height_)
  right_ear_x  = (int)(person.keyPoints[4].position.x  * frame.shape[1]/width_)
  right_ear_y  = (int)(person.keyPoints[4].position.y  * frame.shape[0]/height_)
  left_shoulder_x   = (int)(person.keyPoints[5].position.x  * frame.shape[1]/width_)
  left_shoulder_y   = (int)(person.keyPoints[5].position.y  * frame.shape[0]/height_)
  right_shoulder_x  = (int)(person.keyPoints[6].position.x  * frame.shape[1]/width_)
  right_shoulder_y  = (int)(person.keyPoints[6].position.y  * frame.shape[0]/height_)

  start_point_x = (int) ((left_ear_x + right_ear_x)/2 )
  start_point_y = left_ear_y
  if(right_ear_y < left_ear_y) : start_point_y = right_ear_y

  end_point_x = (int) ((left_shoulder_x + right_shoulder_x)/2 )
  end_point_y = left_shoulder_y
  if(right_shoulder_y > left_shoulder_y) : end_point_y = right_shoulder_y
  cv2.line(frame, (start_point_x, start_point_y) , (end_point_x, end_point_y), (255, 255, 0), 3)


  cv2.imshow('frame', frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()

