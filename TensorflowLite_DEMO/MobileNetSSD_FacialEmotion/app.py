
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


#load model network (MobileNet- Facial Emotion)

# -> facial emotion
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
interpreterFacailEmotion = Interpreter(model_path='facial_expression_detection.tflite')
interpreterFacailEmotion.allocate_tensors() 
input_facial_details = interpreterFacailEmotion.get_input_details()
output_facial_details = interpreterFacailEmotion.get_output_details()
facial_input_width  = input_facial_details[0]['shape'][2]
facial_inpu_height  = input_facial_details[0]['shape'][1]
facial_output_num   = output_facial_details[0]['shape'][1]
# -> load image
facial = cv2.imread("facial_keypoint_image.jpg") #fisrt warm 
facial = cv2.cvtColor(facial, cv2.COLOR_BGR2GRAY)
facial = cv2.resize(facial, (facial_input_width, facial_inpu_height))
# -> interpreter
input_facial_data = np.expand_dims(facial, axis=0)
input_facial_data = np.expand_dims(input_facial_data, axis=3)
interpreterFacailEmotion.set_tensor(input_facial_details[0]['index'], input_facial_data) 
interpreterFacailEmotion.invoke()


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

      #facial emotion
      roi_x0 = getBoardValue(x[0],frame.shape[1])#W
      roi_y0 = getBoardValue(y[0],frame.shape[1])#W
      roi_x1 = getBoardValue(x[1],frame.shape[0])#H
      roi_y1 = getBoardValue(y[1],frame.shape[0])#H
      roi = frame_rgb[ roi_x0 : roi_x1, roi_y0 : roi_y1, :]
      roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
      rot_resized = cv2.resize(roi, (facial_input_width, facial_inpu_height))#rot_resized = np.array(rot_resized, dtype=np.uint8)
      input_facial_data = np.expand_dims(rot_resized, axis=0)
      input_facial_data = np.expand_dims(input_facial_data, axis=3)
      interpreterFacailEmotion.set_tensor(input_facial_details[0]['index'], input_facial_data) 
      interpreterFacailEmotion.invoke()

      #emotion text
      text_x = roi_x0
      text_y = getBoardValue(roi_y0-10,frame.shape[0])#H
      emotion = interpreterFacailEmotion.get_tensor(output_facial_details[0]['index'])#print("keypoint =",keypoint)
      cv2.putText(frame, emotion_dict[np.argmax(emotion)], ( text_x, text_y ), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

  cv2.imshow('frame', frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()

