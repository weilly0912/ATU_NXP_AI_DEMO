
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
# https://google.github.io/mediapipe/solutions/face_mesh.html

# Mediapipe of qaunt tflite need to updat "runtime" version
# May be increase lum of image while using New Web Camera 
# Using "facemesh_weight_flot.tflite" can be accucy result

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

CONTIURS_LEFT_EYE_IDX  = np.array([226, 247, 29, 27, 28, 56, 133, 154, 145, 144, 163, 25])

CONTIURS_RIGHT_EYE_IDX = np.array([413, 441, 257, 259, 359, 390, 373, 374, 380, 463])

CONTIURS_MOUTH_IDX     = np.array([43, 61, 183, 42, 81, 13, 312, 271, 324, 320, 317, 14, 86, 180, 91])

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

# 特徵點判斷是否為左眼輪廓，若是則存進 list 結構中
def IsLeftEyeContoursAndAppend( point_idx, point_x, point_y, list_index, list_data):
    # 左眼輪廓
    if ( point_idx==226 or point_idx==247 or point_idx==29  or point_idx==27  or point_idx==28 or point_idx==56 or point_idx==133 or 
         point_idx==154 or point_idx==145 or point_idx==144 or point_idx==163 or point_idx==25 )  :
         list_index.append(int(point_idx))
         list_data.append([point_x,point_y])

# 特徵點判斷是否為右眼輪廓，若是則存進 list 結構中
def IsRightEyeContoursAndAppend( point_idx, point_x, point_y, list_index, list_data):
    # 右眼輪廓
    if ( point_idx==413 or point_idx==441 or point_idx==257  or point_idx==259  or point_idx==359 or point_idx==390 or point_idx==373 or 
         point_idx==374 or point_idx==380 or point_idx==463)  :
         list_index.append(int(point_idx))
         list_data.append([point_x,point_y])

# 特徵點判斷是否為嘴巴輪廓，若是則存進 list 結構中
def IsMouthContoursAndAppend( point_idx, point_x, point_y, list_index, list_data):
    # 嘴巴輪廓
    if ( point_idx==43  or point_idx==61  or point_idx==183  or point_idx==42  or point_idx==81 or point_idx==13 or point_idx==312 or 
         point_idx==271 or point_idx==324 or point_idx==320  or point_idx==317 or point_idx==14 or point_idx==86 or point_idx==180 or point_idx==91)  :
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
    if (IsFill==0) :
        cv2.drawContours( image, contours, 0, (0,0,0), cv2.FILLED ) 
    else :
        cv2.drawContours( image, contours, 0, (255,255,255), cv2.FILLED ) 

# 取得面具
def get_mask_countours_image( mesh_points, image_shape) :
    idx = 0
    contours_shape_idx    = []
    contours_shape_data   = []
    contours_lefteye_idx  = []
    contours_lefteye_data = []
    contours_righteye_idx = []
    contours_righteye_data= []
    contours_mouth_idx = []
    contours_mouth_data= []            
    # 繪製特徵點
    for pt in mesh_points:
        # 主要特徵點
        x = int(pt[0])
        y = int(pt[1]) 
        IsFaceContoursAndAppend( idx, x, y, contours_shape_idx, contours_shape_data ) # 特徵點判斷是否為臉部輪廓，若是則存進 list 結構中
        IsLeftEyeContoursAndAppend( idx, x, y, contours_lefteye_idx, contours_lefteye_data ) # 特徵點判斷是否為左眼輪廓，若是則存進 list 結構中
        IsRightEyeContoursAndAppend( idx, x, y, contours_righteye_idx, contours_righteye_data ) # 特徵點判斷是否為右眼輪廓，若是則存進 list 結構中
        IsMouthContoursAndAppend( idx, x, y, contours_mouth_idx, contours_mouth_data ) # 特徵點判斷是否為嘴巴輪廓，若是則存進 list 結構中
        idx = idx + 1 # 計數

    # 整理輪廓順序後，繪製輪廓
    mask     = np.zeros(image_shape, dtype="uint8")
    DrawoutContours( mask, CONTIURS_SHAPE_IDX    , contours_shape_idx   , contours_shape_data   , 1) # 繪製外部輪廓
    #DrawoutContours( mask, CONTIURS_LEFT_EYE_IDX , contours_lefteye_idx , contours_lefteye_data , 0) # 左眼繪製輪廓
    #DrawoutContours( mask, CONTIURS_RIGHT_EYE_IDX, contours_righteye_idx, contours_righteye_data, 0) # 右眼繪製輪廓
    #DrawoutContours( mask, CONTIURS_MOUTH_IDX    , contours_mouth_idx   , contours_mouth_data   , 0) # 嘴巴繪製輪廓

    # dilation 
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations = 3)

    return mask

def get_mouth_countours_image( mesh_points, image_shape) :
    idx = 0
    contours_mouth_idx = []
    contours_mouth_data= [] 
    pts_x_max = 0
    pts_y_max = 0
    pts_x_min = 9999
    pts_y_min = 9999    

    # 特徵點判斷是否為嘴巴輪廓，若是則存進 list 結構中
    for pt in mesh_points:
        x = int(pt[0])
        y = int(pt[1]) 
        IsMouthContoursAndAppend( idx, x, y, contours_mouth_idx, contours_mouth_data )
        idx = idx + 1 # 計數

    # 找出最大最小的位置
    for x, y in contours_mouth_data:
        if (x>pts_x_max): pts_x_max=x
        if (y>pts_y_max): pts_y_max=y
        if (x<pts_x_min): pts_x_min=x
        if (y<pts_y_min): pts_y_min=y

    # 整理輪廓順序後，繪製輪廓
    mask     = np.ones(image_shape, dtype="uint8")*255
    DrawoutContours( mask, CONTIURS_MOUTH_IDX    , contours_mouth_idx   , contours_mouth_data   , 0) # 嘴巴繪製輪廓

    pts = [ pts_x_min, pts_y_min, pts_x_max, pts_y_max]

    return mask, pts

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
    APP_NAME = "Facemesh_ChangeFace"
    parser = argparse.ArgumentParser()
    parser.add_argument( '-c' ,"--camera", default="0")
    parser.add_argument("--camera_format", default="V4L2_YUV2_480p")
    parser.add_argument( '-d' ,"--display", default="0")
    parser.add_argument("--save", default="1")
    parser.add_argument( '-t', "--time", default="0")
    parser.add_argument('--delegate' , default="vx", help = 'Please Input vx or xnnpack or ethosu') 
    parser.add_argument("--point_size", default="1")
    parser.add_argument("--IoU", default="0.6")
    parser.add_argument("--model", default="model/mobilenetssd_facedetect_uint8_quant.tflite", help="Using mobilenetssd_facedetect.tflite can be accucy result")
    parser.add_argument("--model_feature", default="model/facemesh_weight_quant.tflite", help="Using facemesh_weight_flot.tflite can be accucy result")
    parser.add_argument("--test_img", default="img/licenseface.jpg")
    parser.add_argument("--offset_y", default="15")
    
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
        
    # 解析解譯器資訊 (人臉位置檢測)
    interpreterFaceExtractor = InferenceDelegate(args.model,args.delegate)
    interpreterFaceExtractor.allocate_tensors() 
    interpreterFaceExtractor_input_details  = interpreterFaceExtractor.get_input_details()
    interpreterFaceExtractor_output_details = interpreterFaceExtractor.get_output_details()
    iFaceExtractor_width    = interpreterFaceExtractor_input_details[0]['shape'][2]
    iFaceExtractor_height   = interpreterFaceExtractor_input_details[0]['shape'][1]
    iFaceExtractor_nChannel = interpreterFaceExtractor_input_details[0]['shape'][3]
    interpreterFaceExtractor.set_tensor(interpreterFaceExtractor_input_details[0]['index'], np.zeros((1,iFaceExtractor_height,iFaceExtractor_width,iFaceExtractor_nChannel)).astype("uint8") ) # 先行進行暖開機
    interpreterFaceExtractor.invoke()

    # 解析解譯器資訊 (面網檢測)
    interpreterFaceMesh = InferenceDelegate(args.model_feature,args.delegate)
    interpreterFaceMesh.allocate_tensors() 
    interpreterFaceMesh_input_details  = interpreterFaceMesh.get_input_details()
    interpreterFaceMesh_output_details = interpreterFaceMesh.get_output_details()
    iFaceMesh_input_width    = interpreterFaceMesh_input_details[0]['shape'][2]
    iFaceMesh_input_height   = interpreterFaceMesh_input_details[0]['shape'][1]
    iFaceMesh_input_nChannel= interpreterFaceMesh_input_details[0]['shape'][3]
    interpreterFaceMesh.set_tensor(interpreterFaceMesh_input_details[0]['index'], np.zeros((1,iFaceMesh_input_height,iFaceMesh_input_width,iFaceMesh_input_nChannel)).astype("float32"))  # 先行進行暖開機
    interpreterFaceMesh.invoke()

    # 預先載入替換的臉孔 (Lena)
    mask_lena = cv2.imread("img/Mask_Lena.jpg")
    mask_lena = cv2.cvtColor(mask_lena, cv2.COLOR_BGR2RGB)
    mask_input_data = cv2.resize(mask_lena, (iFaceMesh_input_width, iFaceMesh_input_height)).astype("float32")
    mask_input_data = (mask_input_data)/255
    mask_input_data = (mask_input_data-0.5)/0.5 # [-0.5,0.5] -> [-1, 1]
    mask_input_data = np.expand_dims(mask_input_data, axis=0)
    interpreterFaceMesh.set_tensor(interpreterFaceMesh_input_details[0]['index'], mask_input_data) 
    interpreterFaceMesh.invoke()
    mesh_mask_lena = interpreterFaceMesh.get_tensor(interpreterFaceMesh_output_details[0]['index']).reshape(468, 3)

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
      interpreterFaceExtractor.invoke()

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
          i = len(scores_nms) # 偵測到第一張人臉就停止

          # --------------------------------------------------------------------------------------------------------
          #  面網檢測  : 將 人臉位置檢測器 所偵測到的人臉逐一檢測
          # --------------------------------------------------------------------------------------------------------
          #  預防邊界
          roi_x0 = max(0, np.floor(x[0] + 0.5).astype('int32'))
          roi_y0 = max(0, np.floor(y[0] + 0.5).astype('int32'))
          roi_x1 = min(frame.shape[1], np.floor(x[1] + 0.5).astype('int32'))
          roi_y1 = min(frame.shape[0], np.floor(y[1] + 0.5).astype('int32'))

          # 設置來源資料至解譯器 取得解譯器的預測結果 (面網檢測) 
          if args.camera =="True" or args.camera == "1" : # 輸入端矯正
              face_img = frame[ roi_y0 : roi_y1, roi_x0 : roi_x1] 
          else :
              face_img = frame_rgb[ roi_y0 : roi_y1, roi_x0 : roi_x1] 

          face_img_resize = cv2.resize(face_img, (iFaceMesh_input_width, iFaceMesh_input_height))
          face_input_data = face_img_resize.astype("float32")
          face_input_data = (face_input_data)/255
          face_input_data = (face_input_data-0.5)/0.5 # [-0.5,0.5] -> [-1, 1]
          face_input_data = np.expand_dims(face_input_data, axis=0)
          interpreterFaceMesh.set_tensor(interpreterFaceMesh_input_details[0]['index'], face_input_data) 
          interpreterFaceMesh.invoke()
          mesh_point = interpreterFaceMesh.get_tensor(interpreterFaceMesh_output_details[0]['index']).reshape(468, 3)

          # 線性映射，找到映射關係並調整大小
          # https://blog.csdn.net/qq_39507748/article/details/104448700
          featurepoint_sample = np.float32([ mesh_mask_lena[10][:2], mesh_mask_lena[234][:2], mesh_mask_lena[152][:2] ])
          featurepoint_images = np.float32([ mesh_point[10][:2]    , mesh_point[234][:2]    , mesh_point[152][:2] ])
          M = cv2.getAffineTransform(featurepoint_sample, featurepoint_images)
          sample_img = cv2.warpAffine(mask_lena,M,(mask_lena.shape[1],mask_lena.shape[0]),5)
 
          # 再次推理 取得面具面網
          sample_img_resize = cv2.resize(sample_img, (iFaceMesh_input_width, iFaceMesh_input_height))
          face_input_data   = sample_img_resize.astype("float32")
          face_input_data   = (face_input_data)/255
          face_input_data   = (face_input_data-0.5)/0.5 # [-0.5,0.5] -> [-1, 1]
          face_input_data   = np.expand_dims(face_input_data, axis=0)
          interpreterFaceMesh.set_tensor(interpreterFaceMesh_input_details[0]['index'], face_input_data) 
          interpreter_time_start = time.time()
          interpreterFaceMesh.invoke()
          interpreter_time_end   = time.time()
          if args.time =="True" or args.time == "1" :
              print( APP_NAME + " Inference Time = ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )
          mesh_point_sample = interpreterFaceMesh.get_tensor(interpreterFaceMesh_output_details[0]['index']).reshape(468, 3)

          # 面具分割圖
          mesh_seg        = get_mask_countours_image( mesh_point       , face_img_resize.shape  ) # 取得原圖面具分割圖
          mesh_seg_sample = get_mask_countours_image( mesh_point_sample, sample_img_resize.shape) # 取得欲替換的面具分割圖
          mask_seg        = np.bitwise_or( mesh_seg, mesh_seg_sample )

          # 找出嘴巴 (填空 位移回去)
          """
          mesh_mouth, pts_mouth  = get_mouth_countours_image( mesh_point , face_img_resize.shape  )
          mesh_mouth_dist_x      = int(mesh_point_sample[320][1] - mesh_point[320][1])
          mesh_mouth_dist_y      = int(mesh_point_sample[152][0] - mesh_point[152][0])
          mask_mouth_seg         = mask_seg.copy()
          mask_mouth_seg[ pts_mouth[1] + mesh_mouth_dist_y : pts_mouth[3] + mesh_mouth_dist_y , \
                          pts_mouth[0] + mesh_mouth_dist_x : pts_mouth[2] + mesh_mouth_dist_x ] = \
                          mesh_mouth[ pts_mouth[1]:pts_mouth[3], pts_mouth[0]:pts_mouth[2] ]
          """

          # 合成面具
          face_mask = (255-mask_seg)*(face_img_resize*255) + ((sample_img_resize*255)*mask_seg) 
          face_mask = face_mask.astype("uint8")
          face_mask = cv2.resize(face_mask, (face_img.shape[1] , face_img.shape[0]))

          # 將面具放置回來源影像中
          frame[ roi_y0 : roi_y1, roi_x0 : roi_x1] = cv2.cvtColor(face_mask, cv2.COLOR_BGR2RGB)
              

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


