from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import time
import numpy as np
import cv2
import keyboard
from tflite_runtime.interpreter import Interpreter

def StyleEncoder( TFLitePath, ImagePath ) :
    # load tflite
    interpreter_style = Interpreter(TFLitePath)
    interpreter_style.allocate_tensors()
    style_input_details  = interpreter_style.get_input_details()
    style_output_details = interpreter_style.get_output_details()
    _, height_predict, width_predict, _ = interpreter_style.get_input_details()[0]['shape']
    
    # load image
    style         = cv2.imread(ImagePath)
    style_rgb     = cv2.cvtColor(style, cv2.COLOR_BGR2RGB) # Acquire frame and resize to expected shape [1xHxWx3]
    style_resized = cv2.resize(style_rgb, (width_predict, height_predict))
    style_resized = np.expand_dims(style_resized, axis=0)
    style_resized = style_resized
    style_resized = np.array(style_resized/255, dtype=np.uint8)

    # learn style from encoder
    interpreter_style.set_tensor(style_input_details[0]["index"], style_resized)
    interpreter_style.invoke()
    style_bottleneck = interpreter_style.get_tensor(style_output_details[0]['index'])
    style_bottleneck = np.array(style_bottleneck, dtype=np.float32)
    #show
    print(" Style Image : ", ImagePath , " is keep !!")

    return style_bottleneck

def StyleTrasfer( frame, width, height, style_bottleneck, interpreter_transfer, transfer_input_details, transfer_output_details):
    #frame_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Acquire frame and resize to expected shape [1xHxWx3]
    frame_resized = cv2.resize(frame, (width, height))
    frame_resized = np.expand_dims(frame_resized, axis=0)
    frame_resized = np.array(frame_resized, dtype=np.float32)
    frame_resized = frame_resized/255
    interpreter_transfer.set_tensor(transfer_input_details[0]["index"], frame_resized)
    interpreter_transfer.set_tensor(transfer_input_details[1]["index"], style_bottleneck)
    interpreter_transfer.invoke()
    stylized_image = interpreter_transfer.get_tensor(transfer_output_details[0]['index'])
    return stylized_image

def main():

  # learn style from encoder
  style_bottleneck_style23   = StyleEncoder("arbitrary_style_transfer_predict.tflite", "style23.jpg")
  style_bottleneck = style_bottleneck_style23 

  # load tflite of Transfer decoder
  interpreter_transfer = Interpreter("magenta_arbitrary-image-stylization-v1-256_int8_transfer_1.tflite")
  interpreter_transfer.allocate_tensors()
  transfer_input_details  = interpreter_transfer.get_input_details()
  transfer_output_details = interpreter_transfer.get_output_details()
  _, height, width, _ = interpreter_transfer.get_input_details()[0]['shape']

  #cap = cv2.VideoCapture(1)
  #cap = cv2.VideoCapture("v4l2src device=/dev/video1 ! video/x-raw,format=YUY2,width=1280,height=720,framerate=30/1! videoscale!videoconvert ! appsink")
  while(True):

    update = input('Updata Frame from Your Camera input [y or n]:')
    #ret, frame = cap.read() 
    frame = cv2.imread("belfry.jpg")
    if update =="y" :
      #start transfer style from decoder
      Start = time.time()
      stylized_image= StyleTrasfer( frame, width, height, style_bottleneck, interpreter_transfer, transfer_input_details, transfer_output_details)
      End = time.time()
      result = stylized_image[0]
      result = cv2.resize(result, (frame.shape[1], frame.shape[0]))
      print("Inference Time = ", End -Start)
    else :
      result = frame

    cv2.imshow('result', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  cv2.destroyAllWindows()
  
if __name__ == "__main__":
    main()
 