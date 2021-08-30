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
  # load tflite of Transfer decoder
  interpreter_transfer = Interpreter("fast_style_transfer.tflite")
  transfer_input_details  = interpreter_transfer.get_input_details()
  transfer_output_details = interpreter_transfer.get_output_details()
  interpreter_transfer.allocate_tensors()
  _, height, width, _ = interpreter_transfer.get_input_details()[0]['shape']
  print(interpreter_transfer.get_input_details())

  frame = cv2.imread("belfry.jpg")
  frame_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  frame_resized = cv2.resize(frame_rgb, (width, height))
  frame_resized = np.expand_dims(frame_resized, axis=0)
  frame_resized = np.array(frame_resized, dtype=np.uint8)
  #frame_resized = frame_resized/255

  interpreter_transfer.set_tensor(transfer_input_details[0]["index"], frame_resized)
  interpreter_transfer.invoke()
  stylized_image = interpreter_transfer.get_tensor(transfer_output_details[0]['index'])

  print("done!!")
  
  
if __name__ == "__main__":
    main()
 