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
# https://github.com/onnx/models/tree/master/vision/body_analysis/emotion_ferplus

import sys
import cv2
import time
import argparse
import numpy as np
from tflite_runtime.interpreter import Interpreter 

# --------------------------------------------------------------------------------------------------------------
# Define
# --------------------------------------------------------------------------------------------------------------
emotion_dict = {0: "neutral", 1: "happiness", 2: "surprise", 3: "sadness", 4: "anger", 5: "disgust", 6: "fear", 7: "contempt"}

# --------------------------------------------------------------------------------------------------------------
# 主程式
# --------------------------------------------------------------------------------------------------------------
def main():

    # 解析外部資訊
    APP_NAME = "Facemesh"
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="0")
    parser.add_argument("--display", default="0")
    parser.add_argument("--save", default="1")
    parser.add_argument("--time", default="0")
    parser.add_argument("--test_img", default="Yichan.jpg")
    parser.add_argument("--offset_y", default="0")
    args = parser.parse_args()

    # 解析解譯器資訊 (人臉位置檢測)
    interpreterFaceExtractor = Interpreter(model_path='mobilenetssd_uint8_face.tflite')
    interpreterFaceExtractor.allocate_tensors() 
    interpreterFaceExtractor_input_details  = interpreterFaceExtractor.get_input_details()
    interpreterFaceExtractor_output_details = interpreterFaceExtractor.get_output_details()
    iFaceExtractor_width    = interpreterFaceExtractor_input_details[0]['shape'][2]
    iFaceExtractor_height   = interpreterFaceExtractor_input_details[0]['shape'][1]
    iFaceExtractor_nChannel = interpreterFaceExtractor_input_details[0]['shape'][3]
    interpreterFaceExtractor.set_tensor(interpreterFaceExtractor_input_details[0]['index'], np.zeros((1,iFaceExtractor_height,iFaceExtractor_width,iFaceExtractor_nChannel)).astype("uint8") ) # 先行進行暖開機
    interpreterFaceExtractor.invoke()

    # 解析解譯器資訊 (表情檢測)
    interpreterFacialEmotion = Interpreter(model_path='emotion-ferplus_uint8.tflite')
    interpreterFacialEmotion.allocate_tensors() 
    interpreterFacialEmotion_input_details  = interpreterFacialEmotion.get_input_details()
    interpreterFacialEmotion_output_details = interpreterFacialEmotion.get_output_details()
    iFacialEmotion_input_width    = interpreterFacialEmotion_input_details[0]['shape'][3]
    iFacialEmotion_input_height   = interpreterFacialEmotion_input_details[0]['shape'][2]
    interpreterFacialEmotion.set_tensor(interpreterFacialEmotion_input_details[0]['index'], np.zeros(( 1, 1, iFacialEmotion_input_height, iFacialEmotion_input_width)).astype("uint8"))  # 先行進行暖開機
    interpreterFacialEmotion.invoke()

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
          x = detection_boxes[0, i, [1, 3]] * frame_rgb.shape[1]
          y = detection_boxes[0, i, [0, 2]] * frame_rgb.shape[0]
          y[1] = y[1] + int(args.offset_y)  #offset

          # 框出偵測到的物件
          rectangle = [x[0], y[0], x[1], y[1]]
          cv2.rectangle(frame, (x[0], y[0]), (x[1], y[1]), (0, 255, 0), 2)

          # --------------------------------------------------------------------------------------------------------
          #  表情檢測  : 將 人臉位置檢測器 所偵測到的人臉逐一檢測
          # --------------------------------------------------------------------------------------------------------
          #  預防邊界
          roi_x0 = max(0, np.floor(x[0] + 0.5).astype('int32'))
          roi_y0 = max(0, np.floor(y[0] + 0.5).astype('int32'))
          roi_x1 = min(frame.shape[1], np.floor(x[1] + 0.5).astype('int32'))
          roi_y1 = min(frame.shape[0], np.floor(y[1] + 0.5).astype('int32'))


          # 設置來源資料至解譯器 (表情檢測)

          if args.camera =="True" or args.camera == "1" : # 輸入端矯正
              gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          else :
              gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)

          face_img = gray[ roi_y0 : roi_y1, roi_x0 : roi_x1] 
          face_img_resized = cv2.resize( face_img, (iFacialEmotion_input_width, iFacialEmotion_input_height))
          face_input_data = np.expand_dims(face_img_resized.astype("uint8"), axis=0)
          face_input_data = np.expand_dims(face_input_data, axis=1)
          interpreterFacialEmotion.set_tensor(interpreterFacialEmotion_input_details[0]['index'], face_input_data) 

          # 解譯器進行推理 (表情檢測)
          interpreter_time_start = time.time()
          interpreterFacialEmotion.invoke()
          interpreter_time_end   = time.time()
          if args.time =="True" or args.time == "1" :
              print( APP_NAME + " Inference Time = ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )

          # 取得解譯器的預測結果 (表情檢測)
          emotion = interpreterFacialEmotion.get_tensor(interpreterFacialEmotion_output_details[0]['index'])

          # 建立輸出結果 (表情檢測)
          text_x = roi_x0
          text_y = min(np.floor( roi_y0 + 0.5 ).astype('int32'), frame.shape[0])
          cv2.putText(frame, emotion_dict[np.argmax(emotion)], ( text_x, text_y ), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

      # 顯示輸出結果
      if args.save == "True" or args.save == "1" :
          cv2.imwrite( APP_NAME + "-" + args.test_img[:len(args.test_img)-4] +'_result.jpg', frame.astype("uint8"))
          print("Save Reuslt Image Success , " + APP_NAME + '_result.jpg')

      if args.display =="True" or args.display == "1" :
          cv2.imshow('frame', frame.astype('uint8'))
          if cv2.waitKey(1) & 0xFF == ord('q'): break

      if args.camera =="False" or args.camera == "0" : sys.exit()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

