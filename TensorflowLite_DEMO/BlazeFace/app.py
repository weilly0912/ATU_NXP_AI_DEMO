# WPI Confidential Proprietary
#--------------------------------------------------------------------------------------
# Copyright (c) 2020 Freescale Semiconductor
# Copyright 2020 WPI
# All Rights Reserved
##--------------------------------------------------------------------------------------
# * Code Ver : 1.0
# * Code Date: 2021/7/30
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
# https://gist.github.com/tworuler/bd7bd4c6cd9a8fbbeb060e7b64cfa008
# https://github.com/ibaiGorordo/BlazeFace-TFLite-Inference

import sys
import cv2
import time
import argparse
import numpy as np
from tflite_runtime.interpreter import Interpreter 
from blazeFaceUtils import gen_anchors, SsdAnchorsCalculatorOptions

# --------------------------------------------------------------------------------------------------------------
# Define
# --------------------------------------------------------------------------------------------------------------
KEY_POINT_SIZE = 6

# --------------------------------------------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------------------------------------------
def nms(boxes, scores, Nt):
    if len(boxes) == 0:
        return [], []
    bboxes = np.array(boxes)
    
    # 計算 n 個候選窗的面積大小
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # 進行排序 (默認從小到大排序)
    order = np.argsort(scores)
 
    picked_boxes   = []  
    picked_scores  = []  
    while order.size > 0:
        # 加入將當前最大的信心度之數值
        index = order[-1]
        picked_boxes.append(boxes[index])
        picked_scores.append(scores[index])
        
        # 獲得當前信心度候選窗與其他候選窗的相交面積
        x11 = np.maximum(x1[index], x1[order[:-1]])
        y11 = np.maximum(y1[index], y1[order[:-1]])
        x22 = np.minimum(x2[index], x2[order[:-1]])
        y22 = np.minimum(y2[index], y2[order[:-1]])
        w = np.maximum(0.0, x22 - x11 + 1)
        h = np.maximum(0.0, y22 - y11 + 1)
        intersection = w * h
        
        # 利用相交的面积和两个框自身的面积计算框的交并比, 将交并比大于阈值的框删除
        ious = intersection / (areas[index] + areas[order[:-1]] - intersection)
        left = np.where(ious < Nt)
        order = order[left]

    # 轉 numpy
    picked_boxes  = np.array(picked_boxes)
    picked_scores = np.array(picked_scores)

    return picked_boxes, picked_scores

# --------------------------------------------------------------------------------------------------------------
# 主程式
# --------------------------------------------------------------------------------------------------------------
def main():

    # 解析外部資訊
    APP_NAME = "BlazeFace"
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="0")
    parser.add_argument("--display", default="0")
    parser.add_argument("--save", default="1")
    parser.add_argument("--time", default="0")
    parser.add_argument("--IoU", default="0.6")
    parser.add_argument("--test_img", default="blazeface_image.jpg")
    parser.add_argument("--score_threshold", default="0.5")
    args = parser.parse_args()

    # 解析解譯器資訊
    interpreter    = Interpreter(model_path='blazeface.tflite')
    interpreter.allocate_tensors() 
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    width    = input_details[0]['shape'][2]
    height   = input_details[0]['shape'][1]
    nChannel = input_details[0]['shape'][3]
    
    # 先行進行暖開機
    interpreter.set_tensor(input_details[0]['index'], np.zeros((1,height,width,nChannel)).astype("float32") )
    interpreter.invoke()

    # 定義資訊(重要)
    ssd_anchors_calculator_options = SsdAnchorsCalculatorOptions(input_size_width=128, input_size_height=128, min_scale=0.1484375, max_scale=0.75
                        , anchor_offset_x=0.5, anchor_offset_y=0.5, num_layers=4
                        , feature_map_width=[], feature_map_height=[]
                        , strides=[8, 16, 16, 16], aspect_ratios=[1.0]
                        , reduce_boxes_in_lowest_layer=False, interpolated_scale_aspect_ratio=1.0
                        , fixed_anchor_size=True)

    anchors = gen_anchors(ssd_anchors_calculator_options)
    score_threshold =float(args.score_threshold)
    sigmoidScoreThreshold = np.log(score_threshold/(1-score_threshold))

    # 是否啟用攝鏡頭
    if args.camera =="True" or args.camera == "1" :
        cap = cv2.VideoCapture("v4l2src device=/dev/video3 ! video/x-raw, format=YUY2, width=1280, height=720, framerate=30/1 ! videoscale!videoconvert ! appsink")
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
          frame = frame +50
          frame_resized = cv2.resize(frame, (width, height))

      else : 
          frame         = cv2.imread(args.test_img)
          frame_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          frame_resized = cv2.resize(frame_rgb, (width, height))
    
      # 設置來源資料至解譯器
      input_data = frame_resized.astype("float32")
      input_data = input_data / 255
      input_data = (input_data - 0.5)/0.5
      input_data = np.expand_dims(input_data, axis=0)
      interpreter.set_tensor(input_details[0]['index'], input_data) 

      # 解譯器進行推理
      interpreter_time_start = time.time()
      interpreter.invoke()
      interpreter_time_end   = time.time()
      if args.time =="True" or args.time == "1" :
          print( APP_NAME + " Inference Time = ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )

      # 取得解譯器的預測結果
      output_feature = np.squeeze(interpreter.get_tensor(output_details[1]['index']))
      output_scores  = np.squeeze(interpreter.get_tensor(output_details[0]['index']))

      # 篩選資料與預測分數
      goodDetections = np.where(output_scores > sigmoidScoreThreshold)[0]
      scores = 1.0 /(1.0 + np.exp(-output_scores[goodDetections]))      

      # 整理輸出
      keypoints = np.zeros((goodDetections.shape[0], KEY_POINT_SIZE, 2))
      boxes = np.zeros((goodDetections.shape[0], 4))
      
      for idx, detectionIdx in enumerate(goodDetections):
          # 解析數據
          anchor = anchors[detectionIdx]
          sx = output_feature[detectionIdx,0]
          sy = output_feature[detectionIdx,1]
          w  = output_feature[detectionIdx,2]
          h  = output_feature[detectionIdx,3]
          
          # 人臉位置解析
          cx = sx + anchor.x_center * width
          cy = sy + anchor.y_center * height
          cx /= width
          cy /= height
          w /= width
          h /= height
          boxes[idx,:] = np.array([cx - w * 0.5, cy - h  , cx + w * 0.5, cy + h ])
          
          # 特徵點解析
          for j in range(KEY_POINT_SIZE):
              lx = output_feature[detectionIdx, 4 + (2 * j) + 0]
              ly = output_feature[detectionIdx, 4 + (2 * j) + 1]
              lx += anchor.x_center * width
              ly += anchor.y_center * height
              lx /= width
              ly /= height
              keypoints[idx,j,:] = np.array([lx, ly])
    

      # 建立輸出結果 
      for i in range(len(scores)):
          
          # 框出人臉
          x0 = int( boxes[i][0]*frame.shape[1] )
          y0 = int( boxes[i][1]*frame.shape[0] )
          x1 = int( boxes[i][2]*frame.shape[1] )
          y1 = int( boxes[i][3]*frame.shape[0] )
          cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 250, 0), 2)          
          
          # 畫出特徵值
          for keypoint in keypoints[i]:
              xKeypoint = (keypoint[0] * frame.shape[1]).astype(int)
              yKeypoint = (keypoint[1] * frame.shape[0]).astype(int)
              cv2.circle(frame,(xKeypoint,yKeypoint), 4, (214, 202, 18), 3)      

      # 顯示輸出結果
      if args.save == "True" or args.save == "1" :
          cv2.imwrite( APP_NAME + "-" + args.test_img[:len(args.test_img)-4] +'_result.jpg', frame.astype("uint8"))
          print("Save Reuslt Image Success , " + APP_NAME + '_result.jpg')

      if args.display =="True" or args.display == "1" :
          cv2.imshow('frame', frame.astype('uint8'))
          if cv2.waitKey(1) & 0xFF == ord('q'): break

      if (args.display =="False" or args.display == "0") and( args.camera =="False" or args.camera == "0" ) : sys.exit()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()