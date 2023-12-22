# WPI Confidential Proprietary
#--------------------------------------------------------------------------------------
# Copyright (c) 2020 Freescale Semiconductor
# Copyright 2020 WPI
# All Rights Reserved
##--------------------------------------------------------------------------------------
# * Code Ver : 4.0
# * Code Date: 2023/04/26
# * Author   : Weilly Li
#--------------------------------------------------------------------------------------
# THIS SOFTWARE IS PROVIDED BY WPI-TW "AS IS" AND ANY EXPRESSED OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL WPI OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
#--------------------------------------------------------------------------------------
# References:
# https://github.com/guichristmann/edge-tpu-tiny-yolo

import os
import re
import sys
import cv2
import time
import argparse
import numpy as np
import colorsys
import random
import tflite_runtime.interpreter as tflite
from utils import *

# --------------------------------------------------------------------------------------------------------------
# Define
# --------------------------------------------------------------------------------------------------------------
V4L2_YUV2_480p = "v4l2src device=/dev/video3 ! video/x-raw,format=YUY2,width=640,height=480, pixel-aspect-ratio=1/1, framerate=30/1! videoscale!videoconvert ! appsink" 
V4L2_YUV2_720p = "v4l2src device=/dev/video3 ! video/x-raw,format=YUY2,width=1280,height=720, pixel-aspect-ratio=1/1, framerate=30/1! videoscale!videoconvert ! appsink"                           
V4L2_H264_1080p = "v4l2src device=/dev/video3 ! video/x-h264, width=1920, height=1080, pixel-aspect-ratio=1/1, framerate=30/1 ! queue ! h264parse ! vpudec ! queue ! queue leaky=1 ! videoscale ! videoconvert ! appsink"

# --------------------------------------------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------------------------------------------
def load_labels(path):
  p = re.compile(r'\s*(\d+)(.+)')
  with open(path, 'r', encoding='utf-8') as f:
    lines = (p.match(line).groups() for line in f.readlines())
  return {int(num): text.strip() for num, text in lines}

def InferenceDelegate( model, delegate ):
    if (delegate=="vx") :
        interpreter = tflite.Interpreter(model, experimental_delegates=[ tflite.load_delegate("/usr/lib/libvx_delegate.so") ])
    elif(delegate=="ethosu"):
        interpreter = tflite.Interpreter(model, experimental_delegates=[tflite.load_delegate("/usr/lib/libethosu_delegate.so")])
    elif(delegate=="xnnpack"):
        interpreter = tflite.Interpreter(model)
    else :
        print("ERROR : Deleget Input Fault")
        return 0
    return interpreter
    
# --------------------------------------------------------------------------------------------------------------
# 主程式
# --------------------------------------------------------------------------------------------------------------
def main():

    # 取得外部輸入資訊
    APP_NAME = "YoLov3_ObjectDetector"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = argparse.ArgumentParser()
    parser.add_argument( '-c' ,"--camera", default="0")
    parser.add_argument("--camera_format", default="V4L2_YUV2_480p")
    parser.add_argument( '-d' ,"--display", default="0")
    parser.add_argument("--save", default="1")
    parser.add_argument( '-t', "--time", default="0")  
    parser.add_argument('--delegate' , default="vx", help = 'Please Input vx or xnnpack or ethosu')  
    parser.add_argument( '-m', '--model' , default="model/coco-tiny-v3-relu_quant.tflite", help='File path of .tflite file.')
    parser.add_argument('--model_input_type' , default="uint8")
    parser.add_argument("--anchors" , default="label/coco_labels.txt", help="Anchors file.")
    parser.add_argument('--labels'  , default="label/coco_labels.txt", help='File path of labels file.')
    parser.add_argument('--threshold',default="0.5")
    parser.add_argument('--test_img', default="img/dog.bmp", help='File path of labels file.')
    
    args = parser.parse_args()
    if args.camera_format == "V4L2_YUV2_480p" : camera_format = V4L2_YUV2_480p
    if args.camera_format == "V4L2_YUV2_720p" : camera_format = V4L2_YUV2_720p
    if args.camera_format == "V4L2_H264_1080p" : camera_format = V4L2_H264_1080p
    
    # vela(NPU) 預設路徑修正
    if(args.delegate=="ethosu"): 
        if(args.model[-11:]!='vela.tflite') :
            args.model = args.model[:-7] + '_vela.tflite'

    # 載入標籤
    labels  = load_labels(args.labels)
    anchors = get_anchors(args.anchors)
    colors  = np.random.uniform(30, 255, size=(len(labels), 3))

    # 解析解譯器資訊
    interpreter = InferenceDelegate(args.model,args.delegate)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape    = input_details[0]["shape"]
    width          = input_details[0]['shape'][2]
    height         = input_details[0]['shape'][1]
    nChannel       = input_details[0]['shape'][3]
    
    # 先行進行暖開機
    interpreter.set_tensor(input_details[0]['index'], np.zeros((1,height,width,nChannel)).astype(args.model_input_type) )
    interpreter.invoke()

    # 是否啟用攝鏡頭
    if args.camera =="True" or args.camera == "1" :
        cap = cv2.VideoCapture(camera_format)
        if(cap.isOpened()==False) :
            print( "Open Camera Failure !!")
            sys.exit()
        else :
            print( "Open Camera Success !!")

    # 迴圈 / 重複推理       
    while(True):
      
      # 視訊/影像資料來源
      if args.camera =="True" or args.camera == "1" :
          ret, frame    = cap.read()
          frame_resized = cv2.resize(frame, (width, height))

      else : 
          frame         = cv2.imread(args.test_img)
          frame_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          frame_resized = cv2.resize(frame_rgb, (width, height))

      # 設置來源資料至解譯器
      input_data = np.expand_dims(frame_resized, axis=0)
      interpreter.set_tensor(input_details[0]['index'], input_data.astype(args.model_input_type)) 

      # 解譯器進行推理
      interpreter_time_start = time.time()
      interpreter.invoke()
      interpreter_time_end   = time.time()
      if args.time =="True" or args.time == "1" :
          print( APP_NAME + " Inference Time = ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )


      # 處理輸出
      out1 = interpreter.get_tensor(output_details[0]['index'])
      out2 = interpreter.get_tensor(output_details[1]['index'])
      
      o1_scale, o1_zero = output_details[0]['quantization']
      o2_scale, o2_zero = output_details[1]['quantization']
      out1 = (out1.astype(np.float32) - o1_zero) * o1_scale
      out2 = (out2.astype(np.float32) - o2_zero) * o2_scale

      n_classes = len(labels)
      _boxes1, _scores1, _classes1 = featuresToBoxes(out1, anchors[[3, 4, 5]], n_classes, input_shape, frame.shape, float(args.threshold))
      _boxes2, _scores2, _classes2 = featuresToBoxes(out2, anchors[[1, 2, 3]], n_classes, input_shape, frame.shape, float(args.threshold))

      if _boxes1.shape[0] == 0:
          _boxes1 = np.empty([0, 2, 2])
          _scores1 = np.empty([0,])
          _classes1 = np.empty([0,])

      if _boxes2.shape[0] == 0:
          _boxes2 = np.empty([0, 2, 2])
          _scores2 = np.empty([0,])
          _classes2 = np.empty([0,])

      boxes = np.append(_boxes1, _boxes2, axis=0)
      scores = np.append(_scores1, _scores2, axis=0)
      classes = np.append(_classes1, _classes2, axis=0)
       
      if len(boxes) > 0:
          boxes, scores, classes = nms_boxes(boxes, scores, classes)


      # 建立輸出結果 - 畫框
      i = 0
      for topleft, botright in boxes:
            # Detected class
            cl = int(classes[i])
            color = tuple(map(int, colors[cl])) 
            # Box coordinates
            topleft = (int(topleft[0]), int(topleft[1]))
            botright = (int(botright[0]), int(botright[1]))
            # Draw box and class
            cv2.rectangle(frame, topleft, botright, color, 2)
            textpos = (topleft[0]-2, topleft[1] - 3)
            score = scores[i] * 100
            cl_name = labels[cl+1]
            text = f"{cl_name} ({score:.1f}%)"
            cv2.putText(frame, text, textpos, cv2.FONT_HERSHEY_DUPLEX,0.45, color, 1, cv2.LINE_AA)
            i += 1

      # 顯示輸出結果
      if args.save == "True" or args.save == "1" :
          cv2.imwrite( "output/" + APP_NAME + "-" + args.test_img.split("/")[-1][:-4] +'_result.jpg', frame.astype("uint8"))
          print("Save Reuslt Image Success , " + APP_NAME +  "-" + args.test_img.split("/")[-1][:-4] +'_result.jpg')

      if args.display =="True" or args.display == "1" :
          cv2.imshow('frame', frame.astype('uint8'))
          if cv2.waitKey(1) & 0xFF == ord('q'): break

      if (args.display =="False" or args.display == "0") and( args.camera =="False" or args.camera == "0" ) : sys.exit()

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
 