import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter 
interpreterPoseEstimation = Interpreter(model_path='bodypix.tflite')
'''
bodypix => json to tflite, none shape
INFO: Created TensorFlow Lite delegate for NNAPI.
malloc(): corrupted top size
'''
interpreterPoseEstimation.resize_tensor_input(interpreterPoseEstimation.get_input_details()[0]['index'], (1, 224, 224, 3))
interpreterPoseEstimation.allocate_tensors() 
input_details = interpreterPoseEstimation.get_input_details()
output_details = interpreterPoseEstimation.get_output_details()
width = input_details[0]['shape'][1]
height = input_details[0]['shape'][2]

frame         = cv2.imread("pose_test_image.jpg")
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_resized = cv2.resize(frame_rgb, (width, height))
frame_resized = np.array(frame_resized, dtype=np.float32)
input_data = np.expand_dims(frame_resized, axis=0)
#input_data = input_data.swapaxes(1, 3)

interpreterPoseEstimation.set_tensor(input_details[0]['index'], input_data) 

interpreterPoseEstimation.invoke()

output   = interpreterPoseEstimation.get_tensor(output_details[0]['index'])

print('done!!')

