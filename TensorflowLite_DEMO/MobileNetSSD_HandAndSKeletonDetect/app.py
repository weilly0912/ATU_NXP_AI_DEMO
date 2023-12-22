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
# https://github.com/yeephycho/widerface-to-tfrecord.git
# https://github.com/tensorflow/models.git
# NMS : 
# https://blog.csdn.net/lz867422770/article/details/100019587

import sys
import cv2
import time
import argparse
import numpy as np
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
# 非極大值抑制 (Non Maximum Suppression)
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

    # 解析外部資訊
    APP_NAME = "HandDetector"
    parser = argparse.ArgumentParser()
    parser.add_argument( '-c' ,"--camera", default="0")
    parser.add_argument("--camera_format", default="V4L2_YUV2_480p")
    parser.add_argument( '-d' ,"--display", default="0")
    parser.add_argument("--save", default="1")
    parser.add_argument( '-t', "--time", default="0")
    parser.add_argument('--delegate' , default="vx", help = 'Please Input vx or xnnpack or ethosu') 
    parser.add_argument( '-m', '--model' , default="model/hand_detect_20000_quant.tflite", help='File path of .tflite file.')
    parser.add_argument( '-mf', '--model_feature' , default="model/hand_landmark_new_256x256_integer_quant.tflite", help='File path of .tflite file.')
    parser.add_argument("--IoU", default="0.6")
    parser.add_argument("--test_img", default="img/hand-1.jpg")
    
    args = parser.parse_args()
    if args.camera_format == "V4L2_YUV2_480p" : camera_format = V4L2_YUV2_480p
    if args.camera_format == "V4L2_YUV2_720p" : camera_format = V4L2_YUV2_720p
    if args.camera_format == "V4L2_H264_1080p" : camera_format = V4L2_H264_1080p

    # vela(NPU) 預設路徑修正
    if(args.delegate=="ethosu"): 
        if(args.model[-11:]!='vela.tflite') :
            args.model = args.model[:-7] + '_vela.tflite'
        if(args.model_feature[-11:]!='vela.tflite') :
            args.model_feature = args.model_feature[:-7] + '_vela.tflite'

    # 解析解譯器資訊(偵測手部)
    interpreterHandDetect = InferenceDelegate(args.model,args.delegate)
    interpreterHandDetect.allocate_tensors() 
    iHandDetect_input_details  = interpreterHandDetect.get_input_details()
    iHandDetect_output_details = interpreterHandDetect.get_output_details()
    iHandDetect_width    = iHandDetect_input_details[0]['shape'][2]
    iHandDetect_height   = iHandDetect_input_details[0]['shape'][1]
    iHandDetect_nChannel = iHandDetect_input_details[0]['shape'][3]
    interpreterHandDetect.set_tensor(iHandDetect_input_details[0]['index'], np.zeros(( 1, iHandDetect_height, iHandDetect_width, iHandDetect_nChannel )).astype("uint8") )
    interpreterHandDetect.invoke()

    # 解析解譯器資訊(偵測手骨)
    interpreterHandSkeleton = InferenceDelegate(args.model_feature,args.delegate)
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
          print( APP_NAME + " Inference Time (Hand Detect) = ", (interpreterHandDetect_time_end - interpreterHandDetect_time_start)*1000 , " ms" )

      # 取得解譯器的預測結果
      detection_boxes   = interpreterHandDetect.get_tensor(iHandDetect_output_details[0]['index'])
      detection_classes = interpreterHandDetect.get_tensor(iHandDetect_output_details[1]['index'])
      detection_scores  = interpreterHandDetect.get_tensor(iHandDetect_output_details[2]['index'])
      num_boxes = interpreterHandDetect.get_tensor(iHandDetect_output_details[3]['index'])

      boxs = np.squeeze(detection_boxes)
      scores = np.squeeze(detection_scores)
      boxs_nms, scores_nms = nms(boxs, scores, float(args.IoU))
    
      # 建立輸出結果 (每個手部)
      for i in range( 0, len(scores_nms)-1) : 
        if scores_nms[i] > .5: # 當物件預測分數大於 50% 時

          # 物件位置
          x = boxs_nms[i, [1, 3]] * frame.shape[1]
          y = boxs_nms[i, [0, 2]] * frame.shape[0]

          # 擴大偵測視窗 
          h = y[1]-y[0]
          w = x[1]-x[0]
          y[0] = y[0] - int(h*0.2)
          y[1] = y[1] - int(h*0.1)
          x[0] = x[0] - int(w*0.4)
          x[1] = x[1] + int(w*0.2)
  
          # 框出偵測到的物件
          cv2.rectangle(frame, ( int(x[0]), int(y[0]) ),  ( int(x[1]), int(y[1]) ), (0, 255, 0), 2) # 畫框於原本影像上

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
          hand_img_resize = cv2.resize(hand_img, ( iHandSkeleton_width, iHandSkeleton_height ))
          hand_input_data = hand_img_resize.astype("float32")
          hand_input_data = (hand_input_data/255)
          hand_input_data = np.expand_dims(hand_input_data, axis=0)
          interpreterHandSkeleton.set_tensor(iHandSkeleton_input_details[0]['index'], hand_input_data) 
          

          # 解譯器進行推理
          interpreter_time_start = time.time()
          interpreterHandSkeleton.invoke()
          interpreter_time_end   = time.time()
          if args.time =="True" or args.time == "1" :
              print( APP_NAME + " Inference Time (Hand Skeleton) = ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )
          
          # 取得解譯器的預測結果
          HandSkeletonFeature = interpreterHandSkeleton.get_tensor(iHandSkeleton_output_details[2]['index'])[0].reshape(21, 3)
          HandSkeletDetected  = interpreterHandSkeleton.get_tensor(iHandSkeleton_output_details[0]['index'])[0]
          
          # 建立輸出結果 - 特徵位置
          Px = []
          Py = []
          size_rate = [ hand_img.shape[1]/iHandSkeleton_width, hand_img.shape[0]/iHandSkeleton_height ]
          for pt in HandSkeletonFeature:
              x = roi_x0 + int(pt[0]*size_rate[0]) 
              y = roi_y0 + int(pt[1]*size_rate[1]) 
              Px.append(x)
              Py.append(y)

          # 建立輸出結果
          if (HandSkeletDetected > 0.7) :
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
               
              # 偵測到則結束
              break

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