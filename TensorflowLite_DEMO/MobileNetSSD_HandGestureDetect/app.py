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

import sys
import cv2
import time
import argparse
import numpy as np
from tflite_runtime.interpreter import Interpreter 

# --------------------------------------------------------------------------------------------------------------
# Define
# --------------------------------------------------------------------------------------------------------------
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# --------------------------------------------------------------------------------------------------------------
# 主程式
# --------------------------------------------------------------------------------------------------------------
def main():

    # 解析外部資訊
    APP_NAME = "HandGestureDetector"
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="0")
    parser.add_argument("--display", default="0")
    parser.add_argument("--save", default="1")
    parser.add_argument("--time", default="0")
    parser.add_argument("--test_img", default="HandGestureDetect_C.jpg")
    args = parser.parse_args()

    # 解析解譯器資訊 (手部)
    interpreterHandDetector    = Interpreter(model_path='hand_detect_20000.tflite')
    interpreterHandDetector.allocate_tensors() 
    iHandDetect_input_details  = interpreterHandDetector.get_input_details()
    iHandDetect_output_details = interpreterHandDetector.get_output_details()
    iHandDetect_width    = iHandDetect_input_details[0]['shape'][2]
    iHandDetect_height   = iHandDetect_input_details[0]['shape'][1]
    iHandDetect_nChannel = iHandDetect_input_details[0]['shape'][3]
    interpreterHandDetector.set_tensor(iHandDetect_input_details[0]['index'], np.zeros((1,iHandDetect_height,iHandDetect_width,iHandDetect_nChannel)).astype("uint8") )
    interpreterHandDetector.invoke()# 先行進行暖開機

    # 解析解譯器資訊
    interpreterGestureDetector    = Interpreter(model_path='Gesture_LanguageMNIST.tflite')
    interpreterGestureDetector.allocate_tensors() 
    iGestureDetect_input_details  = interpreterGestureDetector.get_input_details()
    iGestureDetect_output_details = interpreterGestureDetector.get_output_details()
    iGestureDetect_width    = iGestureDetect_input_details[0]['shape'][2]
    iGestureDetect_height   = iGestureDetect_input_details[0]['shape'][1]
    iGestureDetect_nChannel = iGestureDetect_input_details[0]['shape'][3]
    interpreterGestureDetector.set_tensor(iGestureDetect_input_details[0]['index'], np.zeros(( 1, iGestureDetect_width, iGestureDetect_width, iGestureDetect_nChannel)).astype("uint8") )
    interpreterGestureDetector.invoke()# 先行進行暖開機

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
          frame = frame +30
          frame_resized = cv2.resize(frame, (iHandDetect_width, iHandDetect_height))

      else : 
          frame         = cv2.imread(args.test_img)
          frame_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          frame_resized = cv2.resize(frame_rgb, (iHandDetect_width, iHandDetect_height))
    
      # 設置來源資料至解譯器
      input_data = np.expand_dims(frame_resized, axis=0)
      interpreterHandDetector.set_tensor(iHandDetect_input_details[0]['index'], input_data) 

      # 解譯器進行推理
      interpreterHandDetector_time_start = time.time()
      interpreterHandDetector.invoke()
      interpreterHandDetector_time_end   = time.time()
      if args.time =="True" or args.time == "1" :
          print( APP_NAME + " Inference Time = ", (interpreterHandDetector_time_end - interpreterHandDetector_time_start)*1000 , " ms" )

      # 取得解譯器的預測結果
      detection_boxes   = interpreterHandDetector.get_tensor(iHandDetect_output_details[0]['index'])
      detection_classes = interpreterHandDetector.get_tensor(iHandDetect_output_details[1]['index'])
      detection_scores  = interpreterHandDetector.get_tensor(iHandDetect_output_details[2]['index'])
      num_boxes = interpreterHandDetector.get_tensor(iHandDetect_output_details[3]['index'])
      
      # 建立輸出結果 (每個手部)
      for i in range(int(num_boxes)):
        if detection_scores[0, i] > .5: # 當物件預測分數大於 50% 時

          # 物件位置
          x = detection_boxes[0, i, [1, 3]] * frame.shape[1]
          y = detection_boxes[0, i, [0, 2]] * frame.shape[0]
          if ( (x[1]-x[0])/(y[1]-y[0]) > 0.5 and (x[1]-x[0])/(y[1]-y[0]) < 1.5 ) :

            # 框出偵測到的物件
            rectangle = [x[0], y[0], x[1], y[1]]
            cv2.rectangle(frame, (x[0], y[0]), (x[1], y[1]), (0, 255, 0), 2) # 畫框於原本影像上

            # --------------------------------------------------------------------------------------------------------
            # 手勢識別
            # --------------------------------------------------------------------------------------------------------
            #  預防邊界
            roi_x0 = max(0, np.floor(x[0] + 0.5).astype('int32'))
            roi_y0 = max(0, np.floor(y[0] + 0.5).astype('int32'))
            roi_x1 = min(frame.shape[1], np.floor(x[1] + 0.5).astype('int32'))
            roi_y1 = min(frame.shape[0], np.floor(y[1] + 0.5).astype('int32')) 

            # 設置來源資料至解譯器 (面部特徵檢測)
            if args.camera =="True" or args.camera == "1" : # 輸入端矯正
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else :
                gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)

            face_img = gray[ roi_y0 : roi_y1, roi_x0 : roi_x1]
            face_img = cv2.resize(face_img, (iGestureDetect_width, iGestureDetect_height))#rot_resized = np.array(rot_resized, dtype=np.uint8)
            fece_input_data = np.expand_dims(face_img, axis=0)
            fece_input_data = np.expand_dims(fece_input_data, axis=3)
            interpreterGestureDetector.set_tensor(iGestureDetect_input_details[0]['index'], fece_input_data) 

            # 解譯器進行推理
            interpreterGestureDetector_time_start = time.time()
            interpreterGestureDetector.invoke()
            interpreterGestureDetector_time_end   = time.time()
            if args.time =="True" or args.time == "1" :
                print( APP_NAME + " Inference Time = ", (interpreterGestureDetector_time_end - interpreterGestureDetector_time_start)*1000 , " ms" )
                
            # 處理輸出
            predict = interpreterGestureDetector.get_tensor(iGestureDetect_output_details[0]['index'])
            predicted_label = class_names[np.argmax(predict)]
            print("The max possible Gesture Clssification is ",predicted_label)

            # 標註文字
            cv2.putText(frame, str(predicted_label), (20, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)

            #break # 符合一次就跳出

         
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