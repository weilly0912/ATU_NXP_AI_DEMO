
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


#load model network (CNN-> Facial Age/Gender/Ethnicity)
#--> 
gender_dict = {0: "man", 1: "woman"}
ethnicity_dict = { 0:'asian', 1:'indian', 2:'black', 3:'white', 4:'middle eastern'}
facialshape_dict = { 0:'Heart', 1: 'Oblong' , 2:'Oval', 3:'Round', 4:'Square'}

#---> input data
facial = cv2.imread("facial_image.jpg")
facial = cv2.cvtColor(facial, cv2.COLOR_BGR2GRAY)
facial = cv2.resize(facial, (48, 48))
input_facial_data = np.expand_dims(facial, axis=0)
input_facial_data = np.expand_dims(input_facial_data, axis=3)

#---> age 
interpreterAge = Interpreter(model_path='facial_age_detection.tflite')
interpreterAge.allocate_tensors() 
input_age_details  = interpreterAge.get_input_details()
output_age_details = interpreterAge.get_output_details()
#---> gender
interpreterGender = Interpreter(model_path='facial_gender_detection.tflite')
interpreterGender.allocate_tensors() 
input_gender_details  = interpreterGender.get_input_details()
output_gender_details = interpreterGender.get_output_details()
#---> ethnicity
interpreterEthnicity = Interpreter(model_path='facial_ethnicity_detection.tflite')
interpreterEthnicity.allocate_tensors() 
input_ethnicity_details  = interpreterEthnicity.get_input_details()
output_ethnicity_details = interpreterEthnicity.get_output_details()
#---> shape
interpreterShape = Interpreter(model_path='facial_shape_detection.tflite')
interpreterShape.allocate_tensors() 
input_shape_details  = interpreterShape.get_input_details()
output_shape_details = interpreterShape.get_output_details()


interpreterAge.set_tensor(input_age_details[0]['index'], input_facial_data)
interpreterGender.set_tensor(input_gender_details[0]['index'], input_facial_data) 
interpreterEthnicity.set_tensor(input_ethnicity_details[0]['index'], input_facial_data) 
interpreterShape.set_tensor(input_shape_details[0]['index'], input_facial_data) 

interpreterAge.invoke()
interpreterGender.invoke()
interpreterEthnicity.invoke()
interpreterShape.invoke()

#Board process
def getBoardValue(x,bv):
    if(x<0) :
      x = 0
    if(x>bv):
      x
    return int(x)


def draw_text_line(img, point, text_line: str):
    fontScale = 0.7
    thickness = 2
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    text_line = text_line.split("\n")
    baseline_acc = 0
    for i, text in enumerate(text_line):
        if text:
            text_size, baseline = cv2.getTextSize(str(text), fontFace, fontScale, thickness)
            text_loc = (point[0], point[1] + text_size[1])
            
            # draw score value
            cv2.putText(img, str(text), (text_loc[0], text_loc[1] + baseline_acc), fontFace, fontScale,
                        (0, 0, 255), thickness, 8)
            baseline_acc = baseline_acc + int(baseline*3)

    return img

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

      #Facial Age/Gender/Ethnicity
      # --> roi image = facial
      roi_x0 = getBoardValue(x[0],frame.shape[1])#W
      roi_y0 = getBoardValue(y[0],frame.shape[1])#W
      roi_x1 = getBoardValue(x[1],frame.shape[0])#H
      roi_y1 = getBoardValue(y[1],frame.shape[0])#H
      roi = frame_rgb[ roi_x0 : roi_x1, roi_y0 : roi_y1, :]
      roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
      rot_resized = cv2.resize(roi, (48, 48))#rot_resized = np.array(rot_resized, dtype=np.uint8)
      input_facial_data = np.expand_dims(rot_resized, axis=0)
      input_facial_data = np.expand_dims(input_facial_data, axis=3)

      # -->  interpreter
      interpreterAge.set_tensor(input_age_details[0]['index'], input_facial_data)
      interpreterGender.set_tensor(input_gender_details[0]['index'], input_facial_data) 
      interpreterEthnicity.set_tensor(input_ethnicity_details[0]['index'], input_facial_data)
      interpreterShape.set_tensor(input_shape_details[0]['index'], input_facial_data) 
      interpreterAge.invoke()
      interpreterGender.invoke()
      interpreterEthnicity.invoke()
      interpreterShape.invoke()

      AgePredict       = interpreterAge.get_tensor(output_age_details[0]['index'])#print("keypoint =",keypoint)
      GenderPredict    = interpreterGender.get_tensor(output_gender_details[0]['index'])#print("keypoint =",keypoint)
      EthnicityPredict = interpreterEthnicity.get_tensor(output_ethnicity_details[0]['index'])#print("keypoint =",keypoint)
      ShapePredict     = interpreterShape.get_tensor(output_shape_details[0]['index'])#print("keypoint =",keypoint)

      Age       = AgePredict[0]
      gender    = gender_dict[np.argmax(GenderPredict)]
      ethnicity = ethnicity_dict[np.argmax(EthnicityPredict)]
      facialshape = facialshape_dict[np.argmax(ShapePredict)]
      text_info = "Age : " +  str(Age) + " \n"  + "Gender : " + gender + "\n" + "Ethnicity : " + ethnicity  + "\n" + "Facial Shape : " + facialshape

      #emotion text
      text_x = roi_x0
      text_y = getBoardValue(roi_y0-80,frame.shape[0])
      frame = draw_text_line(frame,(text_x, text_y), text_info)

  cv2.imshow('frame', frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()

