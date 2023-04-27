
# WPI Confidential Proprietary
#--------------------------------------------------------------------------------------
# Copyright (c) 2020 Freescale Semiconductor
# Copyright 2020 WPI
# All Rights Reserved
##--------------------------------------------------------------------------------------
# * Code Ver : 4.0
# * Code Date: 2023/04/26
# * Author   : Weilly Li
# * Non-qunt Model*
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
# https://github.com/magenta/magenta/blob/main/magenta/models/arbitrary_image_stylization/README.md
# https://www.tensorflow.org/hub/tutorials/tf2_arbitrary_image_stylization

import io
import sys
import cv2
import time
import numpy as np
import argparse
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

def StyleEncoder( TFLitePath, ImagePath, delegate ) :
    # load tflite
    interpreter_style = InferenceDelegate(TFLitePath , delegate )
    interpreter_style.allocate_tensors()
    style_input_details  = interpreter_style.get_input_details()
    style_output_details = interpreter_style.get_output_details()
    _, height_predict, width_predict, _ = interpreter_style.get_input_details()[0]['shape']
    
    # load image
    style         = cv2.imread(ImagePath)
    style_rgb     = cv2.cvtColor(style, cv2.COLOR_BGR2RGB) # Acquire frame and resize to expected shape [1xHxWx3]
    style_resized = cv2.resize(style_rgb, (width_predict, height_predict))
    style_resized = np.expand_dims(style_resized, axis=0)
    style_resized = np.array(style_resized, dtype=np.float32)
    style_resized = style_resized/255

    # learn style from encoder
    interpreter_style.set_tensor(style_input_details[0]["index"], style_resized)
    interpreter_style.invoke()
    style_bottleneck = interpreter_style.get_tensor(style_output_details[0]['index'])

    # print infomation
    print("-->> Style Image : ", ImagePath , " learned !!")
    print("------------------------------------------------")

    return style_bottleneck

def StyleTrasfer( frame, width, height, style_bottleneck, interpreter_transfer, transfer_input_details, transfer_output_details):
    frame_resized = cv2.resize(frame, (width, height))
    frame_resized = np.expand_dims(frame_resized, axis=0)
    frame_resized = np.array(frame_resized, dtype=np.float32)
    frame_resized = frame_resized/255
    interpreter_transfer.set_tensor(transfer_input_details[0]["index"], frame_resized)
    interpreter_transfer.set_tensor(transfer_input_details[1]["index"], style_bottleneck)
    interpreter_transfer.invoke()
    stylized_image = interpreter_transfer.get_tensor(transfer_output_details[0]['index'])
    return stylized_image

# --------------------------------------------------------------------------------------------------------------
# 主程式
# --------------------------------------------------------------------------------------------------------------
def main():

  # 解析外部資訊
  APP_NAME = "StyleTrasfer"
  parser = argparse.ArgumentParser()
  parser.add_argument( '-c' ,"--camera", default="0")
  parser.add_argument("--camera_format", default="V4L2_YUV2_480p")
  parser.add_argument( '-d' ,"--display", default="0")
  parser.add_argument("--save", default="1")
  parser.add_argument( '-t', "--time", default="0")
  parser.add_argument('--delegate' , default="xnnpack", help = 'Please Input nnapi or xnnpack')
  parser.add_argument( '-m', '--model' , default="magenta_arbitrary-image-stylization-v1-256_int8_transfer_1.tflite", help='File path of .tflite file.')
  parser.add_argument( '-mf', '--model_feature' , default="magenta_arbitrary-image-stylization-v1-256_int8_prediction_1.tflite", help='File path of .tflite file.')
  parser.add_argument("--test_img", default="belfry.jpg")
  
  args = parser.parse_args()
  if args.camera_format == "V4L2_YUV2_480p" : camera_format = V4L2_YUV2_480p
  if args.camera_format == "V4L2_YUV2_720p" : camera_format = V4L2_YUV2_720p
  if args.camera_format == "V4L2_H264_1080p" : camera_format = V4L2_H264_1080p

  # vela(NPU) 路徑修正
  if(args.delegate=="ethosu"): 
    args.model = 'output/' + args.model[:-7] + '_vela.tflite'
    args.model_feature = 'output/' + args.model_feature[:-7] + '_vela.tflite'


  # (1) 讓解譯器學習各種風格
  style_bottleneck_style23   = StyleEncoder(args.model_feature , "StyleDataSets/style23.jpg",args.delegate)
  style_bottleneck_VanGogh   = StyleEncoder(args.model_feature , "StyleDataSets/VanGogh_Star.jpg",args.delegate)
  style_bottleneck_sketch    = StyleEncoder(args.model_feature , "StyleDataSets/sketch.jpg",args.delegate)
  style_bottleneck_waterpaint= StyleEncoder(args.model_feature , "StyleDataSets/waterpaint.jpg",args.delegate)
  style_bottleneck_colpencil = StyleEncoder(args.model_feature , "StyleDataSets/towers_1916_sq.jpg",args.delegate)

  # (2) 調用任一已學習風格的解譯器
  style_bottleneck = style_bottleneck_VanGogh 

  # (3) 解析解譯器資訊
  interpreter_transfer = InferenceDelegate(args.model,args.delegate)
  interpreter_transfer.allocate_tensors()
  transfer_input_details  = interpreter_transfer.get_input_details()
  transfer_output_details = interpreter_transfer.get_output_details()
  _, height, width, _ = interpreter_transfer.get_input_details()[0]['shape']


  # (4) 迴圈 / 重複推理
  if args.camera =="True" or args.camera == "1" :
    cap = cv2.VideoCapture(camera_format)
    if(cap.isOpened()==False) :
      print( "Open Camera Failure !!")
      sys.exit()
    else :
      print( "Open Camera Success !!")

  while(True):

    # 視訊/影像資料來源
    if args.camera =="True" or args.camera == "1" :
      ret, frame = cap.read() 
    else:
      frame = cv2.imread(args.test_img)
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Acquire frame and resize to expected shape [1xHxWx3]

    # 詢問操作者是否繼續轉換?
    update = input('\n Updata Frame from Your loop input [y or n]:')
    if update =="y" :

      style = input('Updata Style from Your Camera input [0-4]: \n'+\
                    ' Num 0 : style23\n'+\
                    ' Num 1 : VanGogh \n'+\
                    ' Num 2 : sketch \n'+\
                    ' Num 3 : waterpaint \n'+\
                    ' Num 4 : colpencil\n' )
      if style=="0" : style_bottleneck = style_bottleneck_style23
      if style=="1" : style_bottleneck = style_bottleneck_VanGogh  
      if style=="2" : style_bottleneck = style_bottleneck_sketch 
      if style=="3" : style_bottleneck = style_bottleneck_waterpaint 
      if style=="4" : style_bottleneck = style_bottleneck_colpencil  

      # 讓解譯器進行推理，並產生出新的風格影像
      interpreter_time_start = time.time()
      stylized_image = StyleTrasfer( frame, width, height, style_bottleneck, interpreter_transfer, transfer_input_details, transfer_output_details)
      interpreter_time_end   = time.time()
      
      if args.time =="True" or args.time == "1" :
        print( APP_NAME + " Inference Time = ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )

      result = stylized_image[0]*255
      result = cv2.resize(result, (frame.shape[1], frame.shape[0]))

    else :
      break

    # 顯示結果
    if args.save == "True" or args.save == "1" :
      cv2.imwrite( APP_NAME + "-" + args.test_img[:len(args.test_img)-4] +'_result.jpg', result.astype("uint8"))
      print("Save Reuslt Image Success , " + APP_NAME + "-" +  args.test_img[:len(args.test_img)-4] + '_result.jpg')

    if args.display =="True" or args.display == "1" :
      cv2.imshow('result', result.astype("uint8"))
      if cv2.waitKey(1) & 0xFF == ord('q'):break

  cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
 