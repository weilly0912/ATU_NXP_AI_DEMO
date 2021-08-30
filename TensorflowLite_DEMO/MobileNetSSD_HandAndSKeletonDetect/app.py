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
# https://github.com/yeephycho/widerface-to-tfrecord.git
# https://github.com/tensorflow/models.git
# NMS : 
# https://blog.csdn.net/lz867422770/article/details/100019587

import sys
import cv2
import time
import argparse
import numpy as np
from tflite_runtime.interpreter import Interpreter 

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
    APP_NAME = "HandDetector"
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="0")
    parser.add_argument("--display", default="0")
    parser.add_argument("--save", default="1")
    parser.add_argument("--time", default="0")
    parser.add_argument("--IoU", default="0.6")
    parser.add_argument("--test_img", default="hand_detect.jpg")
    args = parser.parse_args()

    # 解析解譯器資訊(偵測手部)
    interpreterHandDetect    = Interpreter(model_path='hand_detect_20000.tflite')
    interpreterHandDetect.allocate_tensors() 
    iHandDetect_input_details  = interpreterHandDetect.get_input_details()
    iHandDetect_output_details = interpreterHandDetect.get_output_details()
    iHandDetect_width    = iHandDetect_input_details[0]['shape'][2]
    iHandDetect_height   = iHandDetect_input_details[0]['shape'][1]
    iHandDetect_nChannel = iHandDetect_input_details[0]['shape'][3]
    interpreterHandDetect.set_tensor(iHandDetect_input_details[0]['index'], np.zeros(( 1, iHandDetect_height, iHandDetect_width, iHandDetect_nChannel )).astype("uint8") )
    interpreterHandDetect.invoke()

    # 解析解譯器資訊(偵測手骨)
    interpreterHandSkeleton    = Interpreter(model_path='hand_landmark.tflite')
    interpreterHandSkeleton.allocate_tensors() 
    iHandSkeleton_input_details  = interpreterHandSkeleton.get_input_details()
    iHandSkeleton_output_details = interpreterHandSkeleton.get_output_details()
    iHandSkeleton_width    = iHandSkeleton_input_details[0]['shape'][2]
    iHandSkeleton_height   = iHandSkeleton_input_details[0]['shape'][1]
    iHandSkeleton_nChannel = iHandSkeleton_input_details[0]['shape'][3]
    interpreterHandSkeleton.set_tensor(iHandSkeleton_input_details[0]['index'], np.zeros(( 1, iHandSkeleton_height, iHandSkeleton_width, iHandSkeleton_nChannel )).astype("float32") )
    interpreterHandSkeleton.invoke()

    # 是否啟用攝鏡頭
    if args.camera =="True" or args.camera == "1" :
        cap = cv2.VideoCapture("v4l2src device=/dev/video1 ! video/x-raw,format=YUY2,iHandDetect_width=1280,iHandDetect_height=720,framerate=30/1! videoscale!videoconvert ! appsink")
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
          frame_resized = cv2.resize(frame, (iHandDetect_width, iHandDetect_height))

      else : 
          frame         = cv2.imread(args.test_img)
          frame_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          frame_resized = cv2.resize(frame_rgb, (iHandDetect_width, iHandDetect_height))
    
      # 設置來源資料至解譯器
      input_data = np.expand_dims(frame_resized, axis=0)
      interpreterHandDetect.set_tensor(iHandDetect_input_details[0]['index'], input_data) 

      # 解譯器進行推理
      interpreterHandDetect_time_start = time.time()
      interpreterHandDetect.invoke()
      interpreterHandDetect_time_end   = time.time()
      if args.time =="True" or args.time == "1" :
          print( APP_NAME + " Inference Time = ", (interpreterHandDetect_time_end - interpreterHandDetect_time_start)*1000 , " ms" )

      # 取得解譯器的預測結果
      detection_boxes   = interpreterHandDetect.get_tensor(iHandDetect_output_details[0]['index'])
      detection_classes = interpreterHandDetect.get_tensor(iHandDetect_output_details[1]['index'])
      detection_scores  = interpreterHandDetect.get_tensor(iHandDetect_output_details[2]['index'])
      num_boxes = interpreterHandDetect.get_tensor(iHandDetect_output_details[3]['index'])

      boxs = np.squeeze(detection_boxes)
      scores = np.squeeze(detection_scores)
      boxs_nms, scores_nms = nms(boxs, scores, float(args.IoU))
    
      # 建立輸出結果 (每個手部)
      for i in range(len(scores_nms)): 
        if scores_nms[i] > .5: # 當物件預測分數大於 50% 時

          # 物件位置
          x = boxs_nms[i, [1, 3]] * frame.shape[1]
          y = boxs_nms[i, [0, 2]] * frame.shape[0]

          # 擴大偵測
          w_adj = int((x[1]-x[0])*0.3)
          h_adj = int((y[1]-y[0])*0.15)
          x[0] = x[0] - w_adj
          x[1] = x[1] + w_adj
          y[0] = y[0] - h_adj
          y[1] = y[1] + 0
 
          # 框出偵測到的物件
          rectangle = [x[0], y[0], x[1], y[1]]
          cv2.rectangle(frame, (x[0], y[0]), (x[1], y[1]), (0, 255, 0), 2) # 畫框於原本影像上

          # --------------------------------------------------------------------------------------------------------
          # 手骨識別
          # --------------------------------------------------------------------------------------------------------
          #  預防邊界
          roi_x0 = max(0, np.floor(x[0] + 0.5).astype('int32'))
          roi_y0 = max(0, np.floor(y[0] + 0.5).astype('int32'))
          roi_x1 = min(frame.shape[1], np.floor(x[1] + 0.5).astype('int32'))
          roi_y1 = min(frame.shape[0], np.floor(y[1] + 0.5).astype('int32')) 

          # 設置來源資料至解譯器 (手骨特徵檢測)
          if args.camera =="True" or args.camera == "1" : # 輸入端矯正
              frame_resized = frame
          else :
              frame_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
              frame_resized = frame_rgb

          hand_img = frame_resized[ roi_y0 : roi_y1, roi_x0 : roi_x1]
          hand_img = cv2.resize(hand_img, ( iHandSkeleton_width, iHandSkeleton_height ))#rot_resized = np.array(rot_resized, dtype=np.uint8)
          hand_input_data = hand_img.astype("float32")
          hand_input_data = (hand_input_data/255)
          hand_input_data = np.expand_dims(hand_input_data, axis=0)
          interpreterHandSkeleton.set_tensor(iHandSkeleton_input_details[0]['index'], hand_input_data) 
          

          # 解譯器進行推理
          interpreter_time_start = time.time()
          interpreterHandSkeleton.invoke()
          interpreter_time_end   = time.time()
          if args.time =="True" or args.time == "1" :
              print( APP_NAME + " Inference Time = ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )
          
          # 取得解譯器的預測結果
          HandSkeletonFeature = interpreterHandSkeleton.get_tensor(iHandSkeleton_output_details[0]['index'])[0].reshape(21, 2)
          HandSkeletDetected  = interpreterHandSkeleton.get_tensor(iHandSkeleton_output_details[1]['index'])[0]

          # 建立輸出結果 - 特徵位置
          Px = []
          Py = []
          size_rate = [ frame.shape[1]/iHandSkeleton_width, frame.shape[0]/iHandSkeleton_height ]
          for pt in HandSkeletonFeature:
              x = int(pt[0]*size_rate[0]) 
              y = int(pt[1]*size_rate[1]) 
              Px.append(x)
              Py.append(y)

          # 建立輸出結果
          if (HandSkeletDetected) :
              # 拇指
              cv2.line(frame, (Px[0], Py[0]) , (Px[1], Py[1]) , (0, 255, 0), 3)
              cv2.line(frame, (Px[1], Py[1]) , (Px[2], Py[2]) , (0, 255, 0), 3)
              cv2.line(frame, (Px[2], Py[2]) , (Px[3], Py[3]) , (0, 255, 0), 3)
              cv2.line(frame, (Px[3], Py[3]) , (Px[4], Py[4]) , (0, 255, 0), 3)

              # 食指
              cv2.line(frame, (Px[0], Py[0]) , (Px[5], Py[5]) , (0, 255, 0), 3)
              cv2.line(frame, (Px[5], Py[5]) , (Px[6], Py[6]) , (0, 255, 0), 3)
              cv2.line(frame, (Px[6], Py[6]) , (Px[7], Py[7]) , (0, 255, 0), 3)
              cv2.line(frame, (Px[7], Py[7]) , (Px[8], Py[8]) , (0, 255, 0), 3)

              # 中指
              cv2.line(frame, (Px[5], Py[5])   , (Px[9], Py[9])   , (0, 255, 0), 3)
              cv2.line(frame, (Px[9], Py[9])   , (Px[10], Py[10]) , (0, 255, 0), 3)
              cv2.line(frame, (Px[10], Py[10]) , (Px[11], Py[11]) , (0, 255, 0), 3)
              cv2.line(frame, (Px[11], Py[11]) , (Px[12], Py[12]) , (0, 255, 0), 3)
              
              # 無名指
              cv2.line(frame, (Px[9], Py[9])   , (Px[13], Py[13]) , (0, 255, 0), 3)
              cv2.line(frame, (Px[13], Py[13]) , (Px[14], Py[14]) , (0, 255, 0), 3)
              cv2.line(frame, (Px[14], Py[14]) , (Px[15], Py[15]) , (0, 255, 0), 3)
              cv2.line(frame, (Px[15], Py[15]) , (Px[16], Py[16]) , (0, 255, 0), 3)
              
              # 小指
              cv2.line(frame, (Px[13], Py[13]) , (Px[17], Py[17]) , (0, 255, 0), 3)
              cv2.line(frame, (Px[17], Py[17]) , (Px[18], Py[18]) , (0, 255, 0), 3)
              cv2.line(frame, (Px[18], Py[18]) , (Px[19], Py[19]) , (0, 255, 0), 3)
              cv2.line(frame, (Px[19], Py[19]) , (Px[20], Py[20]) , (0, 255, 0), 3)
              cv2.line(frame, (Px[17], Py[17]) , (Px[0], Py[0])   , (0, 255, 0), 3)
              
              #指節
              for i in range(len(Px)):
                  cv2.circle(frame, ( Px[i] , Py[i] ), 1, (0, 0, 255), 4)

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