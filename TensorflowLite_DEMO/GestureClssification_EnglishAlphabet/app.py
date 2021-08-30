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
# https://www.kaggle.com/madz2000/cnn-using-keras-100-accuracy

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

    # 取得外部輸入資訊
    APP_NAME = "GestureClssification_EnglishAlphabet"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="0")
    parser.add_argument("--display", default="0")
    parser.add_argument("--save", default="1")
    parser.add_argument("--time", default="0")    
    parser.add_argument('--test_img', default="Gesture_A.jpg", help='File path of labels file.')
    args = parser.parse_args()

    # 解析解譯器資訊
    interpreter = Interpreter("Gesture_LanguageMNIST.tflite")
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    width    = input_details[0]['shape'][2]
    height   = input_details[0]['shape'][1]
    nChannel = input_details[0]['shape'][3]

    # 先行進行暖開機
    interpreter.set_tensor(input_details[0]['index'], np.zeros((1,height,width,nChannel)).astype("uint8") )
    interpreter.invoke()

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
          ret, frame    = cap.read()
          frame_resized = cv2.resize(frame, (width, height))

      else : 
          frame         = cv2.imread(args.test_img)
          frame_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          frame_resized = cv2.resize(frame_rgb, (width, height))

      # 設置來源資料至解譯器
      frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
      input_data = np.expand_dims(frame_resized, axis=0)
      input_data = np.expand_dims(input_data   , axis=3)
      interpreter.set_tensor(input_details[0]['index'], input_data) 

      # 解譯器進行推理
      interpreter_time_start = time.time()
      interpreter.invoke()
      interpreter_time_end   = time.time()
      if args.time =="True" or args.time == "1" :
          print( APP_NAME + " Inference Time = ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )


      # 處理輸出
      predict = interpreter.get_tensor(output_details[0]['index'])
      predicted_label = class_names[np.argmax(predict)]
      print("The max possible Gesture Clssification is ",predicted_label)

      # 標註文字
      cv2.putText(frame, str(predicted_label), (20, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)

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
 








# weilly
'''
# -> back ground method
#gbg  = cv2.bgsegm.createBackgroundSubtractorMOG(history=30, nmixtures=20, backgroundRatio=0.3, noiseSigma=0)
#fgbg  = cv2.createBackgroundSubtractorMOG2()
#fgbg  = cv2.BackgroundSubtractorKNN()

while(True):

  #preprocess image
  ret, frame = cap.read()
  fgmask = fgbg.apply(frame)
  fgmask_resized = cv2.resize(fgmask, (width, height))
  fgmask_resized = cv2.dilate(fgmask_resized, np.ones((3,3), np.uint8), iterations = 2)
  fgmask_resized = cv2.cvtColor(fgmask_resized, cv2.COLOR_GRAY2BGR)
  input_data = np.expand_dims(fgmask_resized, axis=0)#input_data = np.expand_dims(input_data, axis=3)
  interpreter.set_tensor(input_details[0]['index'], input_data) 
  interpreter.invoke()

  #predict
  predict = interpreter.get_tensor(output_details[0]['index'])
  predicted_label = class_names[np.argmax(predict)]
  #print(predicted_label)

  #draw on text
  text = str(predicted_label)
  cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)

  #imshow
  fgmask_resized_show = cv2.resize(fgmask_resized, (frame.shape[1], frame.shape[0]))
  image_show = np.concatenate((frame, fgmask_resized_show)) 
  cv2.imshow('frame', image_show)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

'''

