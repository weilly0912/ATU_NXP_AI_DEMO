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

import sys
import cv2
import time
import argparse
import numpy as np
import tflite_runtime.interpreter as tflite

# --------------------------------------------------------------------------------------------------------------
# Define
# --------------------------------------------------------------------------------------------------------------
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

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
    APP_NAME = "HandGestureDetector"
    parser = argparse.ArgumentParser()
    parser.add_argument( '-c' ,"--camera", default="0")
    parser.add_argument("--camera_format", default="V4L2_YUV2_480p")
    parser.add_argument( '-d' ,"--display", default="0")
    parser.add_argument("--save", default="1")
    parser.add_argument( '-t', "--time", default="0")
    parser.add_argument('--delegate' , default="vx", help = 'Please Input vx or xnnpack or ethosu') 
    parser.add_argument( '-m', '--model' , default="model/hand_detect_20000_quant.tflite", help='File path of .tflite file.')
    parser.add_argument( '-mf', '--model_feature' , default="model/hand_detect_20000_quant.tflite", help='File path of .tflite file.')
    parser.add_argument("--IoU", default="0.6")
    parser.add_argument("--test_img", default="img/Gesture_C.jpg")
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
            
    # 解析解譯器資訊 (手部)
    interpreterHandDetector = InferenceDelegate(args.model,args.delegate)
    interpreterHandDetector.allocate_tensors() 
    iHandDetect_input_details  = interpreterHandDetector.get_input_details()
    iHandDetect_output_details = interpreterHandDetector.get_output_details()
    iHandDetect_width    = iHandDetect_input_details[0]['shape'][2]
    iHandDetect_height   = iHandDetect_input_details[0]['shape'][1]
    iHandDetect_nChannel = iHandDetect_input_details[0]['shape'][3]
    interpreterHandDetector.set_tensor(iHandDetect_input_details[0]['index'], np.zeros((1,iHandDetect_height,iHandDetect_width,iHandDetect_nChannel)).astype("uint8") )
    interpreterHandDetector.invoke()# 先行進行暖開機

    # 解析解譯器資訊
    interpreterGestureDetector = InferenceDelegate(args.model_feature,args.delegate)
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
          print( APP_NAME + " Inference Time (Hand Detect)= ", (interpreterHandDetector_time_end - interpreterHandDetector_time_start)*1000 , " ms" )

      # 取得解譯器的預測結果
      detection_boxes   = interpreterHandDetector.get_tensor(iHandDetect_output_details[0]['index'])
      detection_classes = interpreterHandDetector.get_tensor(iHandDetect_output_details[1]['index'])
      detection_scores  = interpreterHandDetector.get_tensor(iHandDetect_output_details[2]['index'])
      num_boxes = interpreterHandDetector.get_tensor(iHandDetect_output_details[3]['index'])
      
      boxs = np.squeeze(detection_boxes)
      scores = np.squeeze(detection_scores)
      boxs_nms, scores_nms = nms(boxs, scores, float(args.IoU))
    
      # 建立輸出結果 (每個手部)
      for i in range( 0, len(scores_nms)-1) :  
        if scores_nms[i] > .5: # 當物件預測分數大於 50% 時

          # 物件位置
          x = boxs_nms[i, [1, 3]] * frame.shape[1]
          y = boxs_nms[i, [0, 2]] * frame.shape[0]
          if ( (x[1]-x[0])/(y[1]-y[0]) > 0.5 and (x[1]-x[0])/(y[1]-y[0]) < 1.5 ) :

            # 框出偵測到的物件
            cv2.rectangle(frame, ( int(x[0]), int(y[0]) ),  ( int(x[1]), int(y[1]) ), (0, 255, 0), 2) # 畫框於原本影像上

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
            face_img = cv2.resize(face_img, (iGestureDetect_width, iGestureDetect_height))
            fece_input_data = np.expand_dims(face_img, axis=0)
            fece_input_data = np.expand_dims(fece_input_data, axis=3)
            interpreterGestureDetector.set_tensor(iGestureDetect_input_details[0]['index'], fece_input_data) 

            # 解譯器進行推理
            interpreterGestureDetector_time_start = time.time()
            interpreterGestureDetector.invoke()
            interpreterGestureDetector_time_end   = time.time()
            if args.time =="True" or args.time == "1" :
                print( APP_NAME + " Inference Time (Gesture) = ", (interpreterGestureDetector_time_end - interpreterGestureDetector_time_start)*1000 , " ms" )
                
            # 處理輸出
            predict = interpreterGestureDetector.get_tensor(iGestureDetect_output_details[0]['index'])
            predicted_label = class_names[np.argmax(predict)]
            print("The max possible Gesture Clssification is ",predicted_label)

            # 標註文字
            cv2.putText(frame, str(predicted_label), (20, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)

            #break # 符合一次就跳出

         
      # 顯示輸出結果
      if args.save == "True" or args.save == "1" :
          cv2.imwrite( "output/" + APP_NAME + "-" + args.test_img.split("/")[-1][:-4] +'_result.jpg', frame.astype("uint8"))
          print("Save Reuslt Image Success , " + APP_NAME + "-" + args.test_img.split("/")[-1][:-4] + '_result.jpg')

      if args.display =="True" or args.display == "1" :
          cv2.imshow('frame', frame.astype('uint8'))
          if cv2.waitKey(1) & 0xFF == ord('q'): break

      if (args.display =="False" or args.display == "0") and( args.camera =="False" or args.camera == "0" ) : sys.exit()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()