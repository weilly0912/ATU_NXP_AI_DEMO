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
# https://google.github.io/mediapipe/solutions/hands.html
# https://github.com/metalwhale/hand_tracking
# https://tfhub.dev/tensorflow/tfjs-model/handskeleton/1/default/1

# Mediapipe of qaunt tflite need to updat "runtime" version

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
    APP_NAME = "HandSkeleton "
    parser = argparse.ArgumentParser()
    parser.add_argument( '-c' ,"--camera", default="0")
    parser.add_argument("--camera_format", default="V4L2_YUV2_480p")
    parser.add_argument( '-d' ,"--display", default="0")
    parser.add_argument("--save", default="1")
    parser.add_argument( '-t', "--time", default="0")
    parser.add_argument('--delegate' , default="vx", help = 'Please Input vx or xnnpack or ethosu') 
    parser.add_argument( '-m', '--model' , default="model/hand_landmark_new_256x256_integer_quant.tflite", help='File path of .tflite file.')
    parser.add_argument("--test_img", default="img/hand_1.jpg")
    
    args = parser.parse_args()
    if args.camera_format == "V4L2_YUV2_480p" : camera_format = V4L2_YUV2_480p
    if args.camera_format == "V4L2_YUV2_720p" : camera_format = V4L2_YUV2_720p
    if args.camera_format == "V4L2_H264_1080p" : camera_format = V4L2_H264_1080p

    # vela(NPU) 預設路徑修正
    if(args.delegate=="ethosu"): 
        if(args.model[-11:]!='vela.tflite') :
            args.model = args.model[:-7] + '_vela.tflite'

    # 解析解譯器資訊
    interpreter = InferenceDelegate(args.model,args.delegate)
    interpreter.allocate_tensors() 
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    width    = input_details[0]['shape'][2]
    height   = input_details[0]['shape'][1]
    nChannel = input_details[0]['shape'][3]
    
    # 先行進行暖開機
    interpreter.set_tensor(input_details[0]['index'], np.zeros((1,height,width,nChannel)).astype("float32") )
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
      input_data = input_data.astype('float32')
      input_data = input_data /255
      interpreter.set_tensor(input_details[0]['index'], input_data) 

      # 解譯器進行推理
      interpreter_time_start = time.time()
      interpreter.invoke()
      interpreter_time_end   = time.time()
      if args.time =="True" or args.time == "1" :
          print( APP_NAME + " Inference Time = ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )

      # 取得解譯器的預測結果
      feature = interpreter.get_tensor(output_details[2]['index'])[0].reshape(21, 3)
      hand_detected  = interpreter.get_tensor(output_details[0]['index'])[0]

      # 建立輸出結果 - 特徵位置
      Px = []
      Py = []
      size_rate = [frame.shape[1]/width, frame.shape[0]/height]
      for pt in feature:
          x = int(pt[0]*size_rate[0]) 
          y = int(pt[1]*size_rate[1]) 
          Px.append(x)
          Py.append(y)

      # 建立輸出結果
      if (hand_detected) :
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
