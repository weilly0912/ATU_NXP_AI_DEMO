# WPI Confidential Proprietary
#--------------------------------------------------------------------------------------
# Copyright (c) 2020 Freescale Semiconductor
# Copyright 2020 WPI
# All Rights Reserved
##--------------------------------------------------------------------------------------
# * Code Ver : 1.0
# * Code Date: 2022/04/22
# * Author   : Weilly Li
# * Modify by DeeplabV3_SegmationObjection Project
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
# https://github.com/tantara/JejuNet
# https://github.com/tensorflow/models/tree/master/research/deeplab
# https://github.com/sayakpaul/Adventures-in-TensorFlow-Lite/blob/master/DeepLabV3/DeepLab_TFLite_COCO.ipynb
#
# Runtime :
# 513x513 => 340 ms
# 256x256 => 77 ms
# 128x128 => 18 ms (失誤率高)

import re
import sys
import cv2
import colorsys
import random
import time
import argparse
import numpy as np
import tflite_runtime.interpreter as tflite

# --------------------------------------------------------------------------------------------------------------
# Define
# --------------------------------------------------------------------------------------------------------------
LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'person', 'bus',
    'dog', 'cat', 'chair', 'cow', 'diningtable', 'car', 'horse', 'motorbike',
    'botton', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

BASE_LABEL_X =  10
BASE_LABEL_Y =  10

V4L2_YUV2_480p = "v4l2src device=/dev/video3 ! video/x-raw,format=YUY2,width_segmatation=640,height_segmatation=480, pixel-aspect-ratio=1/1, framerate=30/1! videoscale!videoconvert ! appsink" 
V4L2_YUV2_720p = "v4l2src device=/dev/video3 ! video/x-raw,format=YUY2,width_segmatation=1280,height_segmatation=720, pixel-aspect-ratio=1/1, framerate=30/1! videoscale!videoconvert ! appsink"                           
V4L2_H264_1080p = "v4l2src device=/dev/video3 ! video/x-h264, width_segmatation=1920, height_segmatation=1080, pixel-aspect-ratio=1/1, framerate=30/1 ! queue ! h264parse ! vpudec ! queue ! queue leaky=1 ! videoscale ! videoconvert ! appsink"

# --------------------------------------------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------------------------------------------
def load_labels(path):
  p = re.compile(r'\s*(\d+)(.+)')
  with open(path, 'r', encoding='utf-8') as f:
    lines = (p.match(line).groups() for line in f.readlines())
  return {int(num): text.strip() for num, text in lines}

def generate_colors(labels):
  hsv_tuples = [(x / len(labels), 1., 1.) 
                for x in range(len(labels))]

  colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
  colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255),int(x[2] * 255)), colors))
  random.seed(10101)
  random.shuffle(colors)
  random.seed(None)
  return colors

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
    ext_delegate = [ tflite.load_delegate("/usr/lib/libvx_delegate.so") ]
    if (delegate=="vx") :
        interpreter = tflite.Interpreter(model, experimental_delegates=ext_delegate)
    elif(delegate=="xnnpack"):
        interpreter = tflite.Interpreter(model)
    else :
        print("ERROR : Deleget Input Fault")
        return 0
    return interpreter

def SegmationObjectionResult(prediction, frame, frame_resized, args):
    # 濾除物件
    prediction[ prediction==1] =0;  prediction[ prediction==2] =0;  prediction[ prediction==3] =0;   prediction[ prediction==4] =0
    prediction[ prediction==5] =0;  prediction[ prediction==6] =0;  prediction[ prediction==8] =0;   prediction[ prediction==9] =0
    prediction[ prediction==10] =0; prediction[ prediction==11] =0; prediction[ prediction==12] =0;  prediction[ prediction==13] =0
    prediction[ prediction==14] =0; prediction[ prediction==16] =0; prediction[ prediction==17] =0;  prediction[ prediction==18] =0
    prediction[ prediction==19] =0; prediction[ prediction==20] =0; 
    prediction[ prediction==7] =12; prediction[ prediction==15] = 5
        
    # 製作彩色影像
    width       = frame_resized.shape[0]
    height      = frame_resized.shape[1]
    seg_image   = label_to_color_image(prediction).astype(np.uint8)
    seg_image   = cv2.resize(seg_image, (frame.shape[1], frame.shape[0]))
    if args.camera =="True" or args.camera == "1" :
        image_result = cv2.addWeighted(frame, 0.6, seg_image, 0.4, 0)
    else :
        image_result = cv2.addWeighted(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 0.6, seg_image, 0.4, 0)
        #image_result = seg_image

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

    return image_result

def ObjectDetectionResult(positions, classes, scores, labels, colors, frame):
    # 建立輸出結果 
    result = []
    for idx, score in enumerate(scores):
        if score > 0.5:
            result.append({'pos': positions[idx], '_id': classes[idx]})

    # 建立輸出結果  - 顯示結果
    for obj in result:
        pos = obj['pos']
        _id = obj['_id']

        if labels[_id]=="car" or labels[_id]=="truck" or labels[_id]=="train" or \
           labels[_id]=="bus" or labels[_id]=="motorcycle" or labels[_id]=="person" :
            # 物件位置
            x1 = int(pos[1] * frame.shape[1])
            x2 = int(pos[3] * frame.shape[1])
            y1 = int(pos[0] * frame.shape[0])
            y2 = int(pos[2] * frame.shape[0])

            top = max(0, np.floor(y1 + 0.5).astype('int32'))
            left = max(0, np.floor(x1 + 0.5).astype('int32'))
            bottom = min(frame.shape[0], np.floor(y2 + 0.5).astype('int32'))
            right = min(frame.shape[1], np.floor(x2 + 0.5).astype('int32'))

            label_size = cv2.getTextSize(labels[_id], cv2.FONT_HERSHEY_COMPLEX, 0.5, 2)[0]
            label_rect_left = int(left - 3)
            label_rect_top = int(top - 3)
            label_rect_right = int(left + 3 + label_size[0])
            label_rect_bottom = int(top - 5 - label_size[1])

            # 框出偵測到的物件
            cv2.rectangle(frame, (left, top), (right, bottom), colors[int(_id) % len(colors)], 6)
            cv2.rectangle(frame, (label_rect_left, label_rect_top),(label_rect_right, label_rect_bottom), colors[int(_id) % len(colors)], -1)
            cv2.putText(frame, labels[_id], (left, int(top - 4)),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255) , 2)
        
    return frame

# --------------------------------------------------------------------------------------------------------------
# 主程式
# --------------------------------------------------------------------------------------------------------------
def main():

    # 解析外部資訊
    APP_NAME = "ADAS_SegDetection"
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="0")
    parser.add_argument("--camera_format", default="V4L2_YUV2_480p")
    parser.add_argument("--display", default="0")
    parser.add_argument("--save", default="1")
    parser.add_argument("--time", default="0")
    parser.add_argument('--delegate' , default="vx", help = 'Please Input nnapi or xnnpack')
    parser.add_argument("--test_img", default="car.jpg")
    
    args = parser.parse_args()
    if args.camera_format == "V4L2_YUV2_480p" : camera_format = V4L2_YUV2_480p
    if args.camera_format == "V4L2_YUV2_720p" : camera_format = V4L2_YUV2_720p
    if args.camera_format == "V4L2_H264_1080p" : camera_format = V4L2_H264_1080p
    
    # 載入標籤
    labels = load_labels("coco_labels.txt")

    # 載入繪圖顏色資訊
    color_byseg = generate_colors(labels)

    # 解析解譯器資訊 (DeeplabV3_SegmationObjection)
    interpreterSegmatation = InferenceDelegate("deeplabv3_mnv2_pascal_train_256x256.tflite",args.delegate)
    interpreterSegmatation.allocate_tensors() 
    input_details_segmatation  = interpreterSegmatation.get_input_details()
    output_details_segmatation = interpreterSegmatation.get_output_details()
    width_segmatation    = input_details_segmatation[0]['shape'][2]
    height_segmatation   = input_details_segmatation[0]['shape'][1]
    nChannel_segmatation = input_details_segmatation[0]['shape'][3]

    # 解析解譯器資訊 (MobileNet_ObjectDetection)
    interpreterObjectDetection = InferenceDelegate("detect.tflite",args.delegate)
    interpreterObjectDetection.allocate_tensors() 
    input_details_objectDetection  = interpreterObjectDetection.get_input_details()
    output_details_objectDetection = interpreterObjectDetection.get_output_details()
    width_objectDetection    = input_details_objectDetection[0]['shape'][2]
    height_objectDetection   = input_details_objectDetection[0]['shape'][1]
    nChannel_objectDetection = input_details_objectDetection[0]['shape'][3]

    # 先行進行暖開機
    interpreterSegmatation.set_tensor(input_details_segmatation[0]['index'], np.zeros((1,height_segmatation,width_segmatation,nChannel_segmatation)).astype("float32") )
    interpreterObjectDetection.set_tensor(input_details_objectDetection[0]['index'], np.zeros((1,height_objectDetection,width_objectDetection,nChannel_objectDetection)).astype("uint8") )
    interpreterSegmatation.invoke()
    interpreterObjectDetection.invoke()

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
            frame_resized_segmatation = cv2.resize(frame, (width_segmatation, height_segmatation))
            frame_resized_objectdetection = cv2.resize(frame, (width_objectDetection, height_objectDetection))
        else : 
            frame         = cv2.imread(args.test_img)
            frame_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized_segmatation = cv2.resize(frame_rgb, (width_segmatation, height_segmatation))
            frame_resized_objectdetection = cv2.resize(frame_rgb, (width_objectDetection, height_objectDetection))

        # 設置來源資料至解譯器 (DeeplabV3_SegmationObjection)
        input_data_segmatation = frame_resized_segmatation.astype("float32")
        input_data_segmatation = (input_data_segmatation-128)/255
        input_data_segmatation = np.expand_dims(input_data_segmatation, axis=0)
        interpreterSegmatation.set_tensor(input_details_segmatation[0]['index'], input_data_segmatation) 
        
        # 設置來源資料至解譯器 (MobileNet_ObjectDetection)
        input_data_objectdetection = np.expand_dims(frame_resized_objectdetection, axis=0)
        interpreterObjectDetection.set_tensor(input_details_objectDetection[0]['index'], input_data_objectdetection) 

        # 解譯器進行推理 (DeeplabV3_SegmationObjection)
        interpreter_time_start = time.time()
        interpreterSegmatation.invoke()
        interpreter_time_end   = time.time()
        if args.time =="True" or args.time == "1" :
            print(" DeeplabV3 Segmation Objection is Inference Time = ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )

        # 解譯器進行推理 (MobileNet_ObjectDetection)
        interpreter_time_start = time.time()
        interpreterObjectDetection.invoke()
        interpreter_time_end   = time.time()
        if args.time =="True" or args.time == "1" :
            print( "  MobileNet Object Detection is Inference Time = ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )

        # 取得解譯器的預測結果 (DeeplabV3_SegmationObjection)
        prediction_segmatation  = interpreterSegmatation.get_tensor(output_details_segmatation[0]['index']) #fPixel = class
        prediction_segmatation  = np.squeeze(prediction_segmatation)

        # 取得解譯器的預測結果 (MobileNet_ObjectDetection)
        positions_objectDetection = np.squeeze(interpreterObjectDetection.get_tensor(output_details_objectDetection[0]['index']))
        classes_objectDetection   = np.squeeze(interpreterObjectDetection.get_tensor(output_details_objectDetection[1]['index']))
        scores_objectDetection    = np.squeeze(interpreterObjectDetection.get_tensor(output_details_objectDetection[2]['index']))

        # 獲取結果 
        image_result = SegmationObjectionResult(prediction_segmatation, frame, frame_resized_segmatation, args)
        image_result = ObjectDetectionResult(positions_objectDetection, classes_objectDetection, scores_objectDetection, labels, color_byseg, image_result)
  
        # 顯示輸出結果
        if args.save == "True" or args.save == "1" :
            cv2.imwrite( APP_NAME + "-" + args.test_img[:len(args.test_img)-4] +'_result.jpg', image_result.astype("uint8"))
            print("Save Reuslt Image Success , " + APP_NAME + "-" +  args.test_img[:len(args.test_img)-4] + '_result.jpg')

        if args.display =="True" or args.display == "1" :

            cv2.imshow('frame', image_result.astype('uint8'))
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        if (args.display =="False" or args.display == "0") and( args.camera =="False" or args.camera == "0" ) : sys.exit()
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()