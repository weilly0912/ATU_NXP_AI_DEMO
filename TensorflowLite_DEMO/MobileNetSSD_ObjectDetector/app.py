# WPI Confidential Proprietary
#--------------------------------------------------------------------------------------
# Copyright (c) 2020 Freescale Semiconductor
# Copyright 2020 WPI
# All Rights Reserved
##--------------------------------------------------------------------------------------
# * Code Ver : 3.0
# * Code Date: 2022/04/08
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
# https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/raspberry_pi

import re
import sys
import cv2
import time
import argparse
import numpy as np
import colorsys
import random
import tflite_runtime.interpreter as tflite

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

def classify_image(interpreter, top_k=1):
  output_details = interpreter.get_output_details()[0]
  output   = np.squeeze(interpreter.get_tensor(output_details['index']))
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)
  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]

def generate_colors(labels):
  hsv_tuples = [(x / len(labels), 1., 1.) 
                for x in range(len(labels))]

  colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
  colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255),int(x[2] * 255)), colors))
  random.seed(10101)
  random.shuffle(colors)
  random.seed(None)
  return colors

def InferenceDelegate( model, delegate ):
    ext_delegate = [ tflite.load_delegate("/usr/lib/libvx_delegate.so") ]
    if (delegate=="vx") :
        interpreter = tflite.Interpreter(model, experimental_delegates=ext_delegate)
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
    APP_NAME = "ObjectDetector"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="0")
    parser.add_argument("--camera_format", default="V4L2_YUV2_480p")
    parser.add_argument("--display", default="0")
    parser.add_argument("--save", default="1")
    parser.add_argument("--time", default="0")    
    parser.add_argument('--delegate' , default="vx", help = 'Please Input nnapi or xnnpack') 
    parser.add_argument('--model'   , default="detect.tflite", help='File path of .tflite file.')
    parser.add_argument('--model_input_type'   , default="uint8")
    parser.add_argument('--labels'  , default="coco_labels.txt", help='File path of labels file.')
    parser.add_argument('--test_img', default="dog.bmp", help='File path of labels file.')
    
    args = parser.parse_args()
    if args.camera_format == "V4L2_YUV2_480p" : camera_format = V4L2_YUV2_480p
    if args.camera_format == "V4L2_YUV2_720p" : camera_format = V4L2_YUV2_720p
    if args.camera_format == "V4L2_H264_1080p" : camera_format = V4L2_H264_1080p

    # 載入標籤
    labels = load_labels(args.labels)

    # 載入繪圖顏色資訊
    colors = generate_colors(labels)

    # 解析解譯器資訊
    interpreter = InferenceDelegate(args.model,args.delegate)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    width    = input_details[0]['shape'][2]
    height   = input_details[0]['shape'][1]
    nChannel = input_details[0]['shape'][3]

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
          #print( APP_NAME + " Inference Time = ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )
          print( "Inference Time = ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )

      # 處理輸出
      positions = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
      classes   = np.squeeze(interpreter.get_tensor(output_details[1]['index']))
      scores    = np.squeeze(interpreter.get_tensor(output_details[2]['index']))
      
      # 建立輸出結果 
      # 建立輸出結果  - 整理成 list
      result = []
      for idx, score in enumerate(scores):
        if score > 0.5:
          result.append({'pos': positions[idx], '_id': classes[idx]})

      # 建立輸出結果  - 顯示結果
      for obj in result:
        pos = obj['pos']
        _id = obj['_id']

        # 物件位置
        x1 = int(pos[1] * frame.shape[1])
        x2 = int(pos[3] * frame.shape[1])
        y1 = int(pos[0] * frame.shape[0])
        y2 = int(pos[2] * frame.shape[0])

        top = max(0, np.floor(y1 + 0.5).astype('int32'))
        left = max(0, np.floor(x1 + 0.5).astype('int32'))
        bottom = min(frame.shape[0], np.floor(y2 + 0.5).astype('int32'))
        right = min(frame.shape[1], np.floor(x2 + 0.5).astype('int32'))

        label_size = cv2.getTextSize(labels[_id], cv2.FONT_HERSHEY_COMPLEX, 0.5, 2)[0]
        label_rect_left = int(left - 3)
        label_rect_top = int(top - 3)
        label_rect_right = int(left + 3 + label_size[0])
        label_rect_bottom = int(top - 5 - label_size[1])

        # 框出偵測到的物件
        cv2.rectangle(frame, (left, top), (right, bottom), colors[int(_id) % len(colors)], 6)
        cv2.rectangle(frame, (label_rect_left, label_rect_top),(label_rect_right, label_rect_bottom), colors[int(_id) % len(colors)], -1)
        cv2.putText(frame, labels[_id], (left, int(top - 4)),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255) , 2)
      
      # 顯示輸出結果
      if args.save == "True" or args.save == "1" :
          cv2.imwrite( APP_NAME + "-" + args.test_img[:len(args.test_img)-4] +'_result.jpg', frame.astype("uint8"))
          print("Save Reuslt Image Success , " + APP_NAME + "-" +  args.test_img[:len(args.test_img)-4] + '_result.jpg')

      if args.display =="True" or args.display == "1" :
          cv2.imshow('frame', frame.astype('uint8'))
          if cv2.waitKey(1) & 0xFF == ord('q'): break

      if (args.display =="False" or args.display == "0") and( args.camera =="False" or args.camera == "0" ) : sys.exit()
      
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
 