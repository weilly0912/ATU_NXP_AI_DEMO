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

def filter_boxes( box_xywh, scores, score_threshold=0.4, input_shape = ([416,416]) ):
    scores_max = np.max(scores,axis=2)
    mask = scores_max >= score_threshold
    mask = np.squeeze(mask)
    boxs = np.squeeze(box_xywh)
    score = np.squeeze(scores)
    class_boxes = []
    pred_conf = []
    for i in (np.where(mask==True)):
      class_boxes.append(boxs[i][:])
      pred_conf.append(score[i][:])
    class_boxes = np.expand_dims(class_boxes, axis=0)
    pred_conf  = np.expand_dims(pred_conf, axis=0)
    box_xy = np.split(class_boxes, (2, 2), axis=-1)[0]
    box_wh = np.split(class_boxes, (2, 2), axis=-1)[2]
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape
    boxes = np.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    return (boxes, pred_conf)

def combined_non_max_suppression( boxes, scores, threshold ):

  # Avoid zero-matrix
  if len(scores[0][0])==0:
      boxes_new = boxes[0][0][:][:]
      scores_new = scores[0][0][:][:]
      return boxes_new, scores_new

  # calcuate area
  x1 = boxes[0, 0, : , 0]
  y1 = boxes[0, 0, : , 1]
  x2 = boxes[0, 0, : , 2]
  y2 = boxes[0, 0, : , 3]
  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  # class number
  class_num = len( scores[ 0, 0, 0, : ] )
  keep = []
  for class_idx in range(class_num) :
    # score order
    order = scores[ 0 , 0 , : , class_idx ].argsort()[::-1]
    i = order[0]  
    # iou
    while order.size > 0 :
      # index
      i = order[0]
      keep.append(i)
      # calculate max & min location
      xx1 = np.maximum(x1[i], x1[order[1:]])
      yy1 = np.maximum(y1[i], y1[order[1:]])
      xx2 = np.minimum(x2[i], x2[order[1:]])
      yy2 = np.minimum(y2[i], y2[order[1:]])
      # calculate max & min width and height
      w = np.maximum(0.0, xx2 - xx1 + 1)
      h = np.maximum(0.0, yy2 - yy1 + 1)
      inter = w * h 
      # calculate iou
      ious = inter / (areas[i] + areas[order[1:]] - inter)
      inds = np.where(ious <= threshold)[0]
      order = order[inds+1] #update, second select
  keep = np.unique(keep)
  boxes_new = boxes[0][0][keep][:]
  scores_new = scores[0][0][keep][:]

  return boxes_new, scores_new

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
    APP_NAME = "YOLOv4_Tiny_ObjectDetector"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = argparse.ArgumentParser()
    parser.add_argument( '-c' ,"--camera", default="0")
    parser.add_argument( '-d' ,"--display", default="0")
    parser.add_argument("--save", default="1")
    parser.add_argument( '-t', "--time", default="0")    
    parser.add_argument('--delegate' , default="vx", help = 'Please Input vx or xnnpack or ethosu') 
    parser.add_argument( '-m', '--model' , default="model/yolov4tiny-416-OmniXR_quant.tflite", help='File path of .tflite file.')
    parser.add_argument('--model_input_type' , default="float32")
    parser.add_argument('--iou_threshold',default="0.45")
    parser.add_argument('--score_threshold',default="0.25")
    parser.add_argument('--test_img', default="img/test01.jpg", help='File path of labels file.')
    
    args = parser.parse_args()

    # vela(NPU) 路徑修正
    if(args.delegate=="ethosu"): args.model = 'output/' + args.model[:-7] + '_vela.tflite'

    # 載入標籤
    classes=['dog','cat','people'] #labels  = load_labels(args.labels),    anchors = get_anchors(args.anchors)
    num_classes = len(classes)

    # 建立顏色分布表
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(0)
    random.shuffle(colors)
    random.seed(None)  
    
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
        cap = cv2.VideoCapture(2)
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
          frame_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          frame_resized = cv2.resize(frame_rgb, (width, height))
          frame_resized = frame_resized.astype("float32")
          frame_resized = frame_resized / 255.

      else : 
          frame         = cv2.imread(args.test_img)
          frame_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          frame_resized = cv2.resize(frame_rgb, (width, height))
          frame_resized = frame_resized.astype("float32")
          frame_resized = frame_resized / 255.

      # 設置來源資料至解譯器
      input_data = np.expand_dims(frame_resized, axis=0)
      interpreter.set_tensor(input_details[0]['index'], input_data.astype(args.model_input_type)) 

      # 解譯器進行推理
      interpreter_time_start = time.time()
      interpreter.invoke()
      interpreter_time_end   = time.time()
      if args.time =="True" or args.time == "1" :
          print( APP_NAME + " Inference Time = ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )


      # 解析輸出
      image_h, image_w, _ = frame.shape
      pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
      boxes, pred_conf = filter_boxes(pred[0], pred[1], 0.25, input_shape=([width, height]))
      boxes, pred_conf = combined_non_max_suppression(boxes, pred_conf, 0.45)
    
      # 框出物件
      for i in range(len(boxes)):
          
          # 物件位置資訊
          coor = boxes[i][:]
          coor[0] = int(coor[0] * image_h)
          coor[2] = int(coor[2] * image_h)
          coor[1] = int(coor[1] * image_w)
          coor[3] = int(coor[3] * image_w)
          
          # 物件機率資訊
          score = pred_conf[i][:]
          class_ind = np.where(score==score.max())[0][0]
          bbox_color = colors[class_ind]
          bbox_thick = int(0.6 * (image_h + image_w) / 300)
          bbox_mess = '%s: %.2f' % (classes[class_ind], score[class_ind])
          c1, c2 = (coor[1].astype(int), coor[0].astype(int)), (coor[3].astype(int), coor[2].astype(int))
          
          # 畫框資訊
          cv2.rectangle(frame, c1, c2, bbox_color, bbox_thick)
          cv2.putText(frame, bbox_mess, (c1[0], (c1[1] + 15)), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

      # 顯示輸出結果
      if args.save == "True" or args.save == "1" :
          cv2.imwrite( "output/" + APP_NAME + "-" + args.test_img.split("/")[-1][:-4] +'_result.jpg', frame.astype("uint8"))
          print("Save Reuslt Image Success , " + APP_NAME + "-" +  args.test_img.split("/")[-1][:-4] + '_result.jpg')

      if args.display =="True" or args.display == "1" :
          cv2.imshow('frame', frame.astype('uint8'))
          if cv2.waitKey(1) & 0xFF == ord('q'): break

      if (args.display =="False" or args.display == "0") and( args.camera =="False" or args.camera == "0" ) : sys.exit()

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
