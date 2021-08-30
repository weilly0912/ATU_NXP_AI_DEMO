
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


#load model network (MobileNet- Mask Detector)
interpreterMaskDetector = Interpreter(model_path='mask_detection_64x64.tflite')
interpreterMaskDetector.allocate_tensors() 
input_mask_details = interpreterMaskDetector.get_input_details()
output_mask_details = interpreterMaskDetector.get_output_details()
mask_width  = input_mask_details[0]['shape'][2]
mask_height = input_mask_details[0]['shape'][1]
mask_warm_img = cv2.imread("mask_detection_image.jpg") #fisrt warm 
mak_warm_imgres = cv2.resize(mask_warm_img, (mask_width, mask_height))
input_mask_data = np.expand_dims(mak_warm_imgres, axis=0)
interpreterMaskDetector.set_tensor(input_mask_details[0]['index'], input_mask_data) 
interpreterMaskDetector.invoke()

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
  for i in range(10):
    if detection_scores[0, i] > .5:
      x = detection_boxes[0, i, [1, 3]] * frame_rgb.shape[1]
      y = detection_boxes[0, i, [0, 2]] * frame_rgb.shape[0]
      rectangle = [x[0], y[0], x[1], y[1]]
      cv2.rectangle(frame, (x[0], y[0]), (x[1], y[1]), (0, 255, 0), 2)

      #mask detector
      roi_x0 = getBoardValue(x[0],frame.shape[1])#W
      roi_y0 = getBoardValue(y[0],frame.shape[1])#W
      roi_x1 = getBoardValue(x[1],frame.shape[0])#H
      roi_y1 = getBoardValue(y[1],frame.shape[0])#H
      roi = frame_rgb[ roi_x0 : roi_x1, roi_y0 : roi_y1, :]
      rot_resized = cv2.resize(roi, (mask_width, mask_height))#rot_resized = np.array(rot_resized, dtype=np.uint8)
      input_mask_data = np.expand_dims(rot_resized, axis=0)
      interpreterMaskDetector.set_tensor(input_mask_details[0]['index'], input_mask_data) 
      interpreterMaskDetector.invoke()
      
      mask_class_predict = interpreterMaskDetector.get_tensor(output_mask_details[0]['index'])
      print("mask_class_predict =",mask_class_predict)

      #class_id = detection_classes[0, i]
      #print("box=",rectangle,'class_id=',class_id)
  

  cv2.imshow('frame', frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()

