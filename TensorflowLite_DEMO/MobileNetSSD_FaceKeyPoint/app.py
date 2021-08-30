
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
# https://www.kaggle.com/karanjakhar/facial-keypoint-detection

import sys
import cv2
import time
import argparse
import numpy as np
from tflite_runtime.interpreter import Interpreter 

# --------------------------------------------------------------------------------------------------------------
# 主程式
# --------------------------------------------------------------------------------------------------------------
def main():

    # 解析外部資訊
    APP_NAME = "FacialKeyPoint"
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="0")
    parser.add_argument("--display", default="0")
    parser.add_argument("--save", default="1")
    parser.add_argument("--time", default="0")
    parser.add_argument("--test_img", default="yingying.jpg")
    args = parser.parse_args()

    # 解析解譯器資訊 (人臉位置檢測)
    interpreterFaceExtractor = Interpreter(model_path='mobilenetssd_facedetect_uint8_quant.tflite')
    interpreterFaceExtractor.allocate_tensors() 
    interpreterFaceExtractor_input_details  = interpreterFaceExtractor.get_input_details()
    interpreterFaceExtractor_output_details = interpreterFaceExtractor.get_output_details()
    iFaceExtractor_width    = interpreterFaceExtractor_input_details[0]['shape'][2]
    iFaceExtractor_height   = interpreterFaceExtractor_input_details[0]['shape'][1]
    iFaceExtractor_nChannel = interpreterFaceExtractor_input_details[0]['shape'][3]
    interpreterFaceExtractor.set_tensor(interpreterFaceExtractor_input_details[0]['index'], np.zeros((1,iFaceExtractor_height,iFaceExtractor_width,iFaceExtractor_nChannel)).astype("uint8") ) # 先行進行暖開機
    interpreterFaceExtractor.invoke()

    # 解析解譯器資訊 (面部特徵檢測)
    interpreterKeyPoint = Interpreter(model_path='facial_keypoint_detection_new.tflite')
    interpreterKeyPoint.allocate_tensors() 
    interpreterKeyPoint_input_details  = interpreterKeyPoint.get_input_details()
    interpreterKeyPoint_output_details = interpreterKeyPoint.get_output_details()
    iKeyPoint_input_width    = interpreterKeyPoint_input_details[0]['shape'][2]
    iKeyPoint_input_height   = interpreterKeyPoint_input_details[0]['shape'][1]
    iKeyPoint_output_nChannel= interpreterKeyPoint_output_details[0]['shape'][1]
    interpreterKeyPoint.set_tensor(interpreterKeyPoint_input_details[0]['index'], np.zeros((1,iKeyPoint_input_height,iKeyPoint_input_width,1)).astype("uint8"))  # 先行進行暖開機
    interpreterKeyPoint.invoke()

    # 是否啟用攝鏡頭
    if args.camera =="True" or args.camera == "1" :
        cap = cv2.VideoCapture("v4l2src device=/dev/video1 ! video/x-raw,format=YUY2,width=1280,height=720,framerate=30/1! videoscale!videoconvert ! appsink")
        if(cap.isOpened()==False) :
            print( "Open Camera Failure !!")
            sys.exit()
        else :
            print( "Open Camera Success !!")

    # 迴圈 / 重複推理   
    while(True):

      # 視訊/影像資料來源
      if args.camera =="True" or args.camera == "1" :
          ret, frame = cap.read()
          frame_resized = cv2.resize(frame, (iFaceExtractor_width, iFaceExtractor_height))

      else : 
          frame         = cv2.imread(args.test_img)
          frame_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          frame_resized = cv2.resize(frame_rgb, (iFaceExtractor_width, iFaceExtractor_height))
    

      # 設置來源資料至解譯器、並進行推理 (人臉位置檢測)
      input_data = np.expand_dims(frame_resized, axis=0)
      interpreterFaceExtractor.set_tensor(interpreterFaceExtractor_input_details[0]['index'], input_data) 
      interpreterFaceExtractor.invoke()

      # 取得解譯器的預測結果 (人臉位置檢測)
      detection_boxes   = interpreterFaceExtractor.get_tensor(interpreterFaceExtractor_output_details[0]['index'])
      detection_classes = interpreterFaceExtractor.get_tensor(interpreterFaceExtractor_output_details[1]['index'])
      detection_scores  = interpreterFaceExtractor.get_tensor(interpreterFaceExtractor_output_details[2]['index'])
      num_boxes         = interpreterFaceExtractor.get_tensor(interpreterFaceExtractor_output_details[3]['index'])
      
      # 建立輸出結果 (每個人臉)
      for i in range(1):
        if detection_scores[0, i] > .5: # 當物件預測分數大於 50% 時

          # 物件位置
          x = detection_boxes[0, i, [1, 3]] * frame.shape[1]
          y = detection_boxes[0, i, [0, 2]] * frame.shape[0]

          # 框出偵測到的物件
          rectangle = [x[0], y[0], x[1], y[1]]
          cv2.rectangle(frame, (x[0], y[0]), (x[1], y[1]), (0, 255, 0), 2)

          # --------------------------------------------------------------------------------------------------------
          #  面部特徵檢測  : 將 人臉位置檢測器 所偵測到的人臉逐一檢測
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
          face_img = cv2.resize(face_img, (iKeyPoint_input_width, iKeyPoint_input_height))#rot_resized = np.array(rot_resized, dtype=np.uint8)
          fece_input_data = np.expand_dims(face_img, axis=0)
          fece_input_data = np.expand_dims(fece_input_data, axis=3)
          interpreterKeyPoint.set_tensor(interpreterKeyPoint_input_details[0]['index'], fece_input_data) 

          # 解譯器進行推理 (面部特徵檢測)
          interpreter_time_start = time.time()
          interpreterKeyPoint.invoke()
          interpreter_time_end   = time.time()
          if args.time =="True" or args.time == "1" :
              print( APP_NAME + " Inference Time = ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )


          # 取得解譯器的預測結果 (面部特徵檢測)
          keypoint = interpreterKeyPoint.get_tensor(interpreterKeyPoint_output_details[0]['index'])

          # 建立輸出結果 (面部特徵檢測)
          # 特徵點
          for i in range(1,(iKeyPoint_output_nChannel+1),2):
                kx = int( roi_x0 + ( keypoint[0][i-1] * ((roi_x1-roi_x0)/iKeyPoint_input_width) ) )
                ky = int( roi_y0 + ( keypoint[0][i]   * ((roi_y1-roi_y0)/iKeyPoint_input_height) ) )
                cv2.circle(frame, (min(kx ,frame.shape[1]), min(ky,frame.shape[0])), 1, (0, 0, 255), 6)
          
          # 左眼位置
          lefteye_x1 = int( min( roi_x0 + ( keypoint[0][14] * ((roi_x1-roi_x0)/iKeyPoint_input_width)  ) ,frame.shape[1]) )
          lefteye_y1 = int( min( roi_y0 + ( keypoint[0][15] * ((roi_y1-roi_y0)/iKeyPoint_input_height) ) ,frame.shape[0]) )
          lefteye_x2 = int( min( roi_x0 + ( keypoint[0][12] * ((roi_x1-roi_x0)/iKeyPoint_input_width)  ) ,frame.shape[1]) )
          lefteye_y2 = int( min( roi_y0 + ( keypoint[0][7]  * ((roi_y1-roi_y0)/iKeyPoint_input_height) ) ,frame.shape[0]) )


          cv2.rectangle(frame, (lefteye_x1, lefteye_y1), (lefteye_x2, lefteye_y2), (255, 0, 0), 2)

          # 右眼位置
          righteye_x1 = int( min( roi_x0 + ( keypoint[0][16] * ((roi_x1-roi_x0)/iKeyPoint_input_width)  ) ,frame.shape[1]) )
          righteye_y1 = int( min( roi_y0 + ( keypoint[0][17] * ((roi_y1-roi_y0)/iKeyPoint_input_height) ) ,frame.shape[0]) )
          righteye_x2 = int( min( roi_x0 + ( keypoint[0][18] * ((roi_x1-roi_x0)/iKeyPoint_input_width)  ) ,frame.shape[1]) )
          righteye_y2 = int( min( roi_y0 + ( keypoint[0][11] * ((roi_y1-roi_y0)/iKeyPoint_input_height) ) ,frame.shape[0]) )
          cv2.rectangle(frame, (righteye_x1, righteye_y1), (righteye_x2, righteye_y2), (255, 0, 0), 2)

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

