
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


#load model network (MobileNet- Facial KeyPoint)
interpreterKeyPoint = Interpreter(model_path='facial_keypoint_detection_new.tflite')
interpreterKeyPoint.allocate_tensors() 
input_kp_details = interpreterKeyPoint.get_input_details()
output_kp_details = interpreterKeyPoint.get_output_details()
kp_input_width  = input_kp_details[0]['shape'][2]
kp_inpu_height  = input_kp_details[0]['shape'][1]
kp_output_num   = output_kp_details[0]['shape'][1]
keypoint_warm_img = cv2.imread("facial_keypoint_image.jpg") #fisrt warm 
keypoint_warm_img = cv2.cvtColor(keypoint_warm_img, cv2.COLOR_BGR2GRAY)
keypoint_warm_imgres = cv2.resize(keypoint_warm_img, (kp_input_width, kp_inpu_height))
input_kp_data = np.expand_dims(keypoint_warm_imgres, axis=0)
input_kp_data = np.expand_dims(input_kp_data, axis=3)
interpreterKeyPoint.set_tensor(input_kp_details[0]['index'], input_kp_data) 
interpreterKeyPoint.invoke()

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
      roi_y0 = getBoardValue(y[0],frame.shape[1])#W
      roi_x1 = getBoardValue(x[1],frame.shape[0])#H
      roi_y1 = getBoardValue(y[1],frame.shape[0])#H

      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      roi = gray[ roi_x0 : roi_x1, roi_y0 : roi_y1]
      rot_resized = cv2.resize(roi, (kp_input_width, kp_inpu_height))#rot_resized = np.array(rot_resized, dtype=np.uint8)
      input_kp_data = np.expand_dims(rot_resized, axis=0)
      input_kp_data = np.expand_dims(input_kp_data, axis=3)
      interpreterKeyPoint.set_tensor(input_kp_details[0]['index'], input_kp_data) 
      interpreterKeyPoint.invoke()
      keypoint = interpreterKeyPoint.get_tensor(output_kp_details[0]['index'])#print("keypoint =",keypoint)
      for i in range(1,(kp_output_num+1),2):
            kx = roi_x0 + ( keypoint[0][i-1] * ((roi_x1-roi_x0)/kp_input_width) )
            ky = roi_y0 + ( keypoint[0][i]   * ((roi_y1-roi_y0)/kp_inpu_height) )
            cv2.circle(frame, (getBoardValue(kx ,frame.shape[1]), getBoardValue(ky,frame.shape[0])), 1, (0, 0, 255), 6)
      
      #left eye
      kpx1 = getBoardValue( roi_x0 + ( keypoint[0][14] * ((roi_x1-roi_x0)/kp_input_width) ) ,frame.shape[1])
      kpy1 = getBoardValue( roi_y0 + ( keypoint[0][15] * ((roi_y1-roi_y0)/kp_inpu_height) ) ,frame.shape[0])
      kpx2 = getBoardValue( roi_x0 + ( keypoint[0][12] * ((roi_x1-roi_x0)/kp_input_width) ) ,frame.shape[1])
      kpy2 = getBoardValue( roi_y0 + ( keypoint[0][7]  * ((roi_y1-roi_y0)/kp_inpu_height) ) ,frame.shape[0])
      cv2.rectangle(frame, (kpx1, kpy1), (kpx2, kpy2), (255, 0, 0), 2)

      #right eye
      kpx1 = getBoardValue( roi_x0 + ( keypoint[0][16] * ((roi_x1-roi_x0)/kp_input_width) ) ,frame.shape[1])
      kpy1 = getBoardValue( roi_y0 + ( keypoint[0][17] * ((roi_y1-roi_y0)/kp_inpu_height) ) ,frame.shape[0])
      kpx2 = getBoardValue( roi_x0 + ( keypoint[0][18] * ((roi_x1-roi_x0)/kp_input_width) ) ,frame.shape[1])
      kpy2 = getBoardValue( roi_y0 + ( keypoint[0][11] * ((roi_y1-roi_y0)/kp_inpu_height) ) ,frame.shape[0])
      cv2.rectangle(frame, (kpx1, kpy1), (kpx2, kpy2), (255, 0, 0), 2)

  cv2.imshow('frame', frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()

