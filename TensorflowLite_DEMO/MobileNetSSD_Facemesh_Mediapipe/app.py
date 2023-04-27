
# WPI Confidential Proprietary
#--------------------------------------------------------------------------------------
# Copyright (c) 2020 Freescale Semiconductor
# Copyright 2020 WPI
# All Rights Reserved
##--------------------------------------------------------------------------------------
# * Code Ver : 4.0
# * Code Date: 2022/04/08
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
# https://google.github.io/mediapipe/solutions/face_mesh.html

# Mediapipe of qaunt tflite need to updat "runtime" version
# May be increase lum of image while using New Web Camera 
# Using "facemesh_weight_flot.tflite" can be accucy result
# ver2.0 update DrawoutContours function

import sys
import cv2
import time
import argparse
import numpy as np
import tflite_runtime.interpreter as tflite

# --------------------------------------------------------------------------------------------------------------
# Define
# --------------------------------------------------------------------------------------------------------------
CONTIURS_SHAPE_IDX = np.array([10,109,67,103,54,21,162,127,234,93,\
                               132,58,172,136,150,149,176,148,152,\
                               377,378,395,394,365,397,367,416,435,\
                               376,352,345,372,368,300,284,332,297,338])

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

def ScalePointSize(w,w_rate):
    # roi width vs point_size
    s_w = int(w/100) 
    if s_w<1 : s_w = 1
    # roi width vs image width
    if w_rate<1 : w_rate=1
    s_r = int(w_rate*2)
    return s_r*s_w

# 特徵點判斷是否為臉部輪廓，若是則存進 list 結構中
def IsFaceContoursAndAppend( point_idx, point_x, point_y, list_index, list_data):
    # 外圍輪廓
    if ( point_idx==10  or point_idx==109 or point_idx==67  or point_idx==103 or point_idx==54  or point_idx==21  or point_idx==162 or
         point_idx==127 or point_idx==234 or point_idx==93  or point_idx==132 or point_idx==58  or point_idx==172 or point_idx==136 or
         point_idx==150 or point_idx==149 or point_idx==176 or point_idx==148 or point_idx==152 or point_idx==377 or point_idx==378 or
         point_idx==395 or point_idx==394 or point_idx==365 or point_idx==397 or point_idx==367 or point_idx==416 or point_idx==435 or
         point_idx==376 or point_idx==352 or point_idx==345 or point_idx==372 or point_idx==368 or point_idx==300 or point_idx==284 or
         point_idx==332 or point_idx==297 or point_idx==338)  :
         list_index.append(int(point_idx))
         list_data.append([point_x,point_y])

# 輸入特徵點，並畫出輪廓
def DrawoutContours( image, contours_correct_indx, contours_list_index, contours_list_data, IsFill):
    # 整理輪廓
    contours = []
    for pt in contours_correct_indx:
        idx = int(np.where(contours_list_index==pt)[0])
        contours.append(contours_list_data[idx])
    contours = np.array([contours]).astype("int32")

    # 繪製輪廓
    zeros = np.zeros((image.shape),dtype=np.uint8)
    mask = cv2.fillPoly(zeros, contours, color=(120,120,120))
    mask_img = 0.3*mask + image
    mask_img[mask_img>=255] = 255
    return mask_img      


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

# --------------------------------------------------------------------------------------------------------------
# 主程式
# --------------------------------------------------------------------------------------------------------------
def main():

    # 解析外部資訊
    APP_NAME = "Facemesh"
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="0")
    parser.add_argument("--camera_format", default="V4L2_YUV2_480p")
    parser.add_argument("--display", default="0")
    parser.add_argument("--save", default="1")
    parser.add_argument("--time", default="0")
    parser.add_argument('--delegate' , default="vx", help = 'Please Input nnapi or xnnpack')
    parser.add_argument("--point_size", default="1")
    parser.add_argument("--IoU", default="0.6")
    parser.add_argument("--model", default="facemesh_weight_int8.tflite", help="Using facemesh_weight_flot.tflite can be accucy result")
    parser.add_argument("--test_img", default="YangMi.jpg")
    parser.add_argument("--offset_y", default="5")
    
    args = parser.parse_args()
    if args.camera_format == "V4L2_YUV2_480p" : camera_format = V4L2_YUV2_480p
    if args.camera_format == "V4L2_YUV2_720p" : camera_format = V4L2_YUV2_720p
    if args.camera_format == "V4L2_H264_1080p" : camera_format = V4L2_H264_1080p

    # 解析解譯器資訊 (人臉位置檢測)
    interpreterFaceExtractor = InferenceDelegate("mobilenetssd_facedetect_uint8_quant.tflite",args.delegate)
    interpreterFaceExtractor.allocate_tensors() 
    interpreterFaceExtractor_input_details  = interpreterFaceExtractor.get_input_details()
    interpreterFaceExtractor_output_details = interpreterFaceExtractor.get_output_details()
    iFaceExtractor_width    = interpreterFaceExtractor_input_details[0]['shape'][2]
    iFaceExtractor_height   = interpreterFaceExtractor_input_details[0]['shape'][1]
    iFaceExtractor_nChannel = interpreterFaceExtractor_input_details[0]['shape'][3]
    interpreterFaceExtractor.set_tensor(interpreterFaceExtractor_input_details[0]['index'], np.zeros((1,iFaceExtractor_height,iFaceExtractor_width,iFaceExtractor_nChannel)).astype("uint8") ) # 先行進行暖開機
    interpreterFaceExtractor.invoke()

    # 解析解譯器資訊 (面網檢測)
    interpreterFaceMesh = InferenceDelegate( args.model, args.delegate )
    interpreterFaceMesh.allocate_tensors() 
    interpreterFaceMesh_input_details  = interpreterFaceMesh.get_input_details()
    interpreterFaceMesh_output_details = interpreterFaceMesh.get_output_details()
    iFaceMesh_input_width    = interpreterFaceMesh_input_details[0]['shape'][2]
    iFaceMesh_input_height   = interpreterFaceMesh_input_details[0]['shape'][1]
    iFaceMesh_input_nChannel= interpreterFaceMesh_input_details[0]['shape'][3]
    interpreterFaceMesh.set_tensor(interpreterFaceMesh_input_details[0]['index'], np.zeros((1,iFaceMesh_input_height,iFaceMesh_input_width,iFaceMesh_input_nChannel)).astype("float32"))  # 先行進行暖開機
    interpreterFaceMesh.invoke()

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
          frame_resized = cv2.resize(frame, (iFaceExtractor_width, iFaceExtractor_height))

      else : 
          frame         = cv2.imread(args.test_img)
          frame_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          frame_resized = cv2.resize(frame_rgb, (iFaceExtractor_width, iFaceExtractor_height))
    
      # 設置來源資料至解譯器、並進行推理 (人臉位置檢測)
      input_data = np.expand_dims(frame_resized, axis=0)
      interpreterFaceExtractor.set_tensor(interpreterFaceExtractor_input_details[0]['index'], input_data) 
      interpreter_time_start = time.time()
      interpreterFaceExtractor.invoke()
      interpreter_time_end   = time.time()
      if args.time =="True" or args.time == "1" :
          print( APP_NAME + " Inference Time (Face Extractor) = ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )

      # 取得解譯器的預測結果 (人臉位置檢測)
      detection_boxes   = interpreterFaceExtractor.get_tensor(interpreterFaceExtractor_output_details[0]['index'])
      detection_classes = interpreterFaceExtractor.get_tensor(interpreterFaceExtractor_output_details[1]['index'])
      detection_scores  = interpreterFaceExtractor.get_tensor(interpreterFaceExtractor_output_details[2]['index'])
      num_boxes         = interpreterFaceExtractor.get_tensor(interpreterFaceExtractor_output_details[3]['index'])

      boxs = np.squeeze(detection_boxes)
      scores = np.squeeze(detection_scores)
      boxs_nms, scores_nms = nms(boxs, scores, float(args.IoU))

      # 建立輸出結果 
      for i in range( 0, len(scores_nms)-1) : 
        if scores_nms[i] > .5: # 當物件預測分數大於 50% 時
        
          # 物件位置
          x = boxs_nms[i, [1, 3]] * frame.shape[1]
          y = boxs_nms[i, [0, 2]] * frame.shape[0]
          x[1] = x[1] + int((x[1]-x[0])*0.1)
          y[1] = y[1] + int((y[1]-y[0])*0.05) + int(args.offset_y)

          # 框出偵測到的物件
          cv2.rectangle(frame, ( int(x[0]), int(y[0]) ),  ( int(x[1]), int(y[1]) ), (0, 255, 0), 2) 

          # --------------------------------------------------------------------------------------------------------
          #  面網檢測  : 將 人臉位置檢測器 所偵測到的人臉逐一檢測
          # --------------------------------------------------------------------------------------------------------
          #  預防邊界
          roi_x0 = max(0, np.floor(x[0] + 0.5).astype('int32'))
          roi_y0 = max(0, np.floor(y[0] + 0.5).astype('int32'))
          roi_x1 = min(frame.shape[1], np.floor(x[1] + 0.5).astype('int32'))
          roi_y1 = min(frame.shape[0], np.floor(y[1] + 0.5).astype('int32'))

          # 設置來源資料至解譯器 (面網檢測)
          if args.camera =="True" or args.camera == "1" : # 輸入端矯正
              face_img = frame[ roi_y0 : roi_y1, roi_x0 : roi_x1] 
          else :
              face_img = frame_rgb[ roi_y0 : roi_y1, roi_x0 : roi_x1] 

          face_input_data = cv2.resize(face_img, (iFaceMesh_input_width, iFaceMesh_input_height)).astype("float32")
          face_input_data = (face_input_data)/255
          face_input_data = (face_input_data-0.5)/0.5 # [-0.5,0.5] -> [-1, 1]
          face_input_data  = np.expand_dims(face_input_data, axis=0)
          interpreterFaceMesh.set_tensor(interpreterFaceMesh_input_details[0]['index'], face_input_data) 

          # 解譯器進行推理 (面網檢測)
          interpreter_time_start = time.time()
          interpreterFaceMesh.invoke()
          interpreter_time_end   = time.time()
          if args.time =="True" or args.time == "1" :
              print( APP_NAME + " Inference Time (Face Mesh) = ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )

          # 取得解譯器的預測結果 (面網檢測)
          mesh = interpreterFaceMesh.get_tensor(interpreterFaceMesh_output_details[0]['index']).reshape(468, 3)
          
          # 比例關係
          size_rate = [face_img.shape[1]/iFaceMesh_input_width, face_img.shape[0]/iFaceMesh_input_height]
          if args.point_size=="0" :
              pointsize = ScalePointSize((roi_x1-roi_x0),size_rate[0])
          else :
              pointsize = int(args.point_size)

          # 建立輸出結果 (面網檢測)
          if(1) : #if(facedetected) :
            image_mask= np.zeros(frame.shape, dtype="uint8")
            idx = 0
            contours_shape_idx    = []
            contours_shape_data   = []
            for pt in mesh:
                x = int(roi_x0 + pt[0]*size_rate[0]) 
                y = int(roi_y0 + pt[1]*size_rate[1]) 
                cv2.circle(frame, ( x ,y ), 1, (200, 200, 200),  pointsize)
                IsFaceContoursAndAppend( idx, x, y, contours_shape_idx, contours_shape_data ) # 特徵點判斷是否為臉部輪廓，若是則存進 list 結構中
                idx = idx + 1

            frame = DrawoutContours( frame, CONTIURS_SHAPE_IDX    , contours_shape_idx   , contours_shape_data   , 1) # 繪製外部輪廓

      # 顯示輸出結果
      if args.save == "True" or args.save == "1" :
          cv2.imwrite( APP_NAME + "-" + args.test_img[:len(args.test_img)-4] +'_result.jpg', frame.astype("uint8"))
          print("Save Reuslt Image Success , " + APP_NAME + "-" +  args.test_img[:len(args.test_img)-4] + '_result.jpg')

      if args.display =="True" or args.display == "1" :
          cv2.imshow('frame', frame.astype('uint8'))
          if cv2.waitKey(1) & 0xFF == ord('q'): break

      if (args.display =="False" or args.display == "0") and( args.camera =="False" or args.camera == "0" ) : sys.exit()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


