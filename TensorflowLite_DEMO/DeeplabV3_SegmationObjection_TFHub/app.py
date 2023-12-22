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
# https://tfhub.dev/
# https://github.com/sayakpaul/Adventures-in-TensorFlow-Lite/blob/master/DeepLabV3/DeepLab_TFLite_COCO.ipynb
# 
# Runtime :
# lite-model_deeplabv3-mobilenetv2_dm05-int8_1_default_1 => 960.8 ms
# lite-model_deeplabv3-mobilenetv2-int8_1_default_1 => 2126.9 ms


import sys
import cv2
import time
import argparse
import numpy as np
import tflite_runtime.interpreter as tflite

# --------------------------------------------------------------------------------------------------------------
# Define
# --------------------------------------------------------------------------------------------------------------
LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

BASE_LABEL_X =  10
BASE_LABEL_Y =  10

V4L2_YUV2_480p = "v4l2src device=/dev/video3 ! video/x-raw,format=YUY2,width=640,height=480, pixel-aspect-ratio=1/1, framerate=30/1! videoscale!videoconvert ! appsink" 
V4L2_YUV2_720p = "v4l2src device=/dev/video3 ! video/x-raw,format=YUY2,width=1280,height=720, pixel-aspect-ratio=1/1, framerate=30/1! videoscale!videoconvert ! appsink"                           
V4L2_H264_1080p = "v4l2src device=/dev/video3 ! video/x-h264, width=1920, height=1080, pixel-aspect-ratio=1/1, framerate=30/1 ! queue ! h264parse ! vpudec ! queue ! queue leaky=1 ! videoscale ! videoconvert ! appsink"

# --------------------------------------------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------------------------------------------
def create_pascal_label_colormap():

  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap

def label_to_color_image(label):

  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

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
    APP_NAME = "SegmationObjection_TF Hub"
    parser = argparse.ArgumentParser()
    parser.add_argument( '-c' ,"--camera", default="0")
    parser.add_argument("--camera_format", default="V4L2_YUV2_480p")
    parser.add_argument( '-d' ,"--display", default="0")
    parser.add_argument("--save", default="1")
    parser.add_argument( '-t', "--time", default="0")
    parser.add_argument('--delegate' , default="vx", help = 'Please Input vx or xnnpack or ethosu') 
    parser.add_argument("--model", default="model/lite-model_deeplabv3-mobilenetv2_dm05-int8_1_default_1_quant.tflite")
    parser.add_argument("--test_img", default="img/dog.png")
    
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
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    width = input_details[0]['shape'][2]
    height = input_details[0]['shape'][1]
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
            ret, frame = cap.read()
            frame_resized = cv2.resize(frame, (width, height))
        else : 
            frame         = cv2.imread(args.test_img)
            frame_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height))

        # 設置來源資料至解譯器
        input_data = frame_resized.astype("float32")
        input_data = (input_data-128)/255
        input_data = np.expand_dims(input_data, axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data) 

        # 解譯器進行推理
        interpreter_time_start = time.time()
        interpreter.invoke()
        interpreter_time_end   = time.time()
        if args.time =="True" or args.time == "1" :
            print( APP_NAME + " Inference Time = ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )

        # 取得解譯器的預測結果
        prediction  = interpreter.get_tensor(output_details[0]['index']) #fPixel = class
        prediction  = np.squeeze(prediction).astype(np.int8)
        prediction  = np.argmax(prediction,axis=2)
        seg_image   = label_to_color_image(prediction).astype(np.uint8)
        seg_image   = cv2.resize(seg_image, (width, height))

        if args.camera =="True" or args.camera == "1" :
            image_result = cv2.addWeighted(frame_resized, 0.7, seg_image, 0.3, 0)
        else :
            image_result = cv2.addWeighted(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB), 0.7, seg_image, 0.3, 0)
 
        # 重新調整輸出影像大小
        image_result = cv2.resize(image_result, (frame.shape[1], frame.shape[0]))

        # 產生標籤影像
        unique_labels = np.unique(prediction)
        unique_labels = unique_labels[1:]

        if( len(unique_labels) >= 1 ) :
            FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
            FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
            image_colormap = FULL_COLOR_MAP[unique_labels].astype("uint8")
            image_colormap = cv2.resize(image_colormap, ( int(frame.shape[1]*0.075), int(frame.shape[0]*0.2)))
            
            # 合併標籤影像至結果圖中
            image_result[ BASE_LABEL_Y: BASE_LABEL_Y + image_colormap.shape[0], BASE_LABEL_X: BASE_LABEL_X + image_colormap.shape[1], :] = image_colormap       
            
            # 框出標籤
            cv2.rectangle(image_result, ( int(BASE_LABEL_X), int(BASE_LABEL_Y) ),  ( int( BASE_LABEL_X + image_colormap.shape[1] ), int( BASE_LABEL_Y + image_colormap.shape[0] ) ), (0, 0, 0), 2) 

            # 標記標籤至結果圖中
            label_x = BASE_LABEL_X + image_colormap.shape[1]
            label_y = BASE_LABEL_Y + int( image_colormap.shape[0] * 0.3 )
            label_y_step = int( image_colormap.shape[0] / len(unique_labels) )
            for idx in unique_labels :
                cv2.putText( image_result,  str(LABEL_NAMES[idx]), ( label_x, label_y ), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv2.LINE_AA)
                label_y = label_y + label_y_step


        # 顯示輸出結果
        if args.save == "True" or args.save == "1" :
            cv2.imwrite( "output/" + APP_NAME + "-" + args.test_img.split("/")[-1][:-4] +'_result.jpg', image_result.astype("uint8"))
            print("Save Reuslt Image Success , " + APP_NAME + "-" +  args.test_img.split("/")[-1][:-4] + '_result.jpg')

        if args.display =="True" or args.display == "1" :
            cv2.imshow('frame', image_result.astype('uint8'))
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        if (args.display =="False" or args.display == "0") and( args.camera =="False" or args.camera == "0" ) : sys.exit()
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()