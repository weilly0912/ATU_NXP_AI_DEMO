from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter

def main():

  # (1) load tflite of Style 
  # Get Width & Height from Interpreter 
  interpreter_style = Interpreter("magenta_arbitrary-image-stylization-v1-256_int8_prediction_1.tflite")
  interpreter_style.allocate_tensors()
  style_input_details  = interpreter_style.get_input_details()
  style_output_details = interpreter_style.get_output_details()
  _, height_predict, width_predict, _ = interpreter_style.get_input_details()[0]['shape']

  # (2) load tflite of Transfer
  # Get Width & Height from Interpreter 
  interpreter_transfer = Interpreter("magenta_arbitrary-image-stylization-v1-256_int8_transfer_1.tflite")
  interpreter_transfer.allocate_tensors()
  transfer_input_details  = interpreter_transfer.get_input_details()
  transfer_output_details = interpreter_transfer.get_output_details()
  _, height, width, _ = interpreter_transfer.get_input_details()[0]['shape']

  #(3) learn style from encoder
  style         = cv2.imread("style23.jpg")
  style_rgb     = cv2.cvtColor(style, cv2.COLOR_BGR2RGB) # Acquire frame and resize to expected shape [1xHxWx3]
  style_resized = cv2.resize(style_rgb, (width_predict, height_predict))
  style_resized = np.expand_dims(style_resized, axis=0)
  style_resized = np.array(style_resized, dtype=np.float32)
  style_resized = style_resized/255
  interpreter_style.set_tensor(style_input_details[0]["index"], style_resized)
  interpreter_style.invoke()
  style_bottleneck = interpreter_style.get_tensor(style_output_details[0]['index'])

  #(4) Loading Image
  frame         = cv2.imread("belfry.jpg")
  frame_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Acquire frame and resize to expected shape [1xHxWx3]
  frame_resized = cv2.resize(frame_rgb, (width, height))
  frame_resized = np.expand_dims(frame_resized, axis=0)
  frame_resized = np.array(frame_resized, dtype=np.float32)
  frame_resized = frame_resized/255
  
  #(5) start transfer style from decoder
  Start = time.time()
  transfer_input_details = interpreter_transfer.get_input_details()
  interpreter_transfer.set_tensor(transfer_input_details[0]["index"], frame_resized)
  interpreter_transfer.set_tensor(transfer_input_details[1]["index"], style_bottleneck)
  interpreter_transfer.invoke()
  stylized_image = interpreter_transfer.get_tensor(transfer_output_details[0]['index'])
  End = time.time()

  while(True):
    print("Inference Time = ", End -Start)
    cv2.imshow('result', stylized_image[0])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  cv2.destroyAllWindows()
  
if __name__ == "__main__":
    main()
 