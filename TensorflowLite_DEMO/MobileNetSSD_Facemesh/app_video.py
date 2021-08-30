
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter 

#load model network (MobileNet SSD- Face Exactor)
interpreterFaceExtractor = Interpreter(model_path='mobilenetssd_uint8_face.tflite')
interpreterFaceExtractor.allocate_tensors() 
input_details = interpreterFaceExtractor.get_input_details()
output_details = interpreterFaceExtractor.get_output_details()
width = input_details[0]['shape'][2]
height = input_details[0]['shape'][1]


#load model network (MobileNet- facemesh)
interpreterFaceMesh = Interpreter(model_path='facemesh.tflite')
interpreterFaceMesh.allocate_tensors() 
facemesh_input_details  = interpreterFaceMesh.get_input_details()
facemesh_output_details = interpreterFaceMesh.get_output_details()
facemesh_width  =  facemesh_input_details[0]['shape'][2]
facemesh_height =  facemesh_input_details[0]['shape'][3]
facemesh_frame  = cv2.imread("face_example.jpg")
frame_rgb = cv2.cvtColor(facemesh_frame, cv2.COLOR_BGR2RGB)
frame_resized = cv2.resize(frame_rgb, (facemesh_width, facemesh_height))
facemesh_input_data = np.expand_dims(frame_resized.astype("float32"), axis=0)#.astype("float32")
facemesh_input_data = facemesh_input_data.swapaxes(1, 3)
interpreterFaceMesh.set_tensor(facemesh_input_details[0]['index'], facemesh_input_data) 
interpreterFaceMesh.invoke()

#Board process
def getBoardValue(x,bv):
    if(x<0) :
      x = 0
    if(x>bv):
      x
    return int(x)

cap = cv2.VideoCapture(1)
while(True):

  ret, frame = cap.read()

  #load image and set tensor
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  frame_resized = cv2.resize(frame_rgb, (width, height))
  input_data = np.expand_dims(frame_resized, axis=0)
  interpreterFaceExtractor.set_tensor(input_details[0]['index'], input_data) 
 
  #start interpreter
  interpreterFaceExtractor.invoke()

  #predict
  detection_boxes = interpreterFaceExtractor.get_tensor(output_details[0]['index'])
  detection_classes = interpreterFaceExtractor.get_tensor(output_details[1]['index'])
  detection_scores = interpreterFaceExtractor.get_tensor(output_details[2]['index'])
  num_boxes = interpreterFaceExtractor.get_tensor(output_details[3]['index'])
  for i in range(1):
    if detection_scores[0, i] > .5:
      x = detection_boxes[0, i, [1, 3]] * frame_rgb.shape[1]
      y = detection_boxes[0, i, [0, 2]] * frame_rgb.shape[0]
      rectangle = [x[0], y[0], x[1], y[1]]
      cv2.rectangle(frame, (x[0], y[0]), (x[1], y[1]), (0, 255, 0), 2)

      
      #keypoint detector
      roi_x0 = getBoardValue(x[0],frame.shape[1])#W
      roi_y0 = getBoardValue(y[0],frame.shape[0])#
      roi_x1 = getBoardValue(x[1],frame.shape[1])#W
      roi_y1 = getBoardValue(y[1],frame.shape[0])#H

      roi = frame[ roi_x0 : roi_x1, roi_y0 : roi_y1, :]
      roi_resized = cv2.resize(roi, (facemesh_width, facemesh_height))#frame_resized = np.array(frame_resized, dtype=np.float32)
      facemesh_input_data = np.expand_dims(roi_resized.astype("float32"), axis=0)#.astype("float32")
      facemesh_input_data = facemesh_input_data.swapaxes(1, 3)
      #print(facemesh_input_data.shape)
      
      interpreterFaceMesh.set_tensor(facemesh_input_details[0]['index'], facemesh_input_data) 
      interpreterFaceMesh.invoke()

      landmarks_result = interpreterFaceMesh.get_tensor(facemesh_output_details[0]['index'])
      landmarks_result = np.reshape(landmarks_result, (-1, 468, 3))[:, :, :2] # landmarks_result.shape[1] = 1404 / 3  

      for land in landmarks_result :
            for pt in land:
              x = int(roi_x0 + pt[0]*roi.shape[0]/192)
              y = int(roi_y0 + pt[1]*roi.shape[1]/192)
              cv2.circle(frame, ( x , y ), 1, (0, 0, 255), 2)

  cv2.imshow('frame', frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()

