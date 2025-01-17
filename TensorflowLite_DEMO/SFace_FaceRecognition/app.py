
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
# https://www.kaggle.com/karanjakhar/facial-keypoint-detection

import os
import sys
import cv2
import time
import argparse
import numpy as np
import tflite_runtime.interpreter as tflite

# --------------------------------------------------------------------------------------------------------------
# Define
# --------------------------------------------------------------------------------------------------------------
V4L2_YUV2_480p = "v4l2src device=/dev/video2 ! video/x-raw,format=YUY2,width=640,height=480, pixel-aspect-ratio=1/1, framerate=30/1! videoscale!videoconvert ! appsink" 
V4L2_YUV2_720p = "v4l2src device=/dev/video2 ! video/x-raw,format=YUY2,width=1280,height=720, pixel-aspect-ratio=1/1, framerate=30/1! videoscale!videoconvert ! appsink"                           
V4L2_H264_1080p = "v4l2src device=/dev/video2 ! video/x-h264, width=1920, height=1080, pixel-aspect-ratio=1/1, framerate=30/1 ! queue ! h264parse ! vpudec ! queue ! queue leaky=1 ! videoscale ! videoconvert ! appsink"

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

def cos_similarity(X, Y):
    Y = Y.T
    # (128,) x (n, 128) = (n,)
    result = np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y, axis=0))
    return result

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
    APP_NAME = "FaceRecongition"
    parser = argparse.ArgumentParser()
    parser.add_argument( '-c' ,"--camera", default="0")
    parser.add_argument("--camera_format", default="V4L2_YUV2_480p")
    parser.add_argument( '-d' ,"--display", default="0")
    parser.add_argument("--save", default="1")
    parser.add_argument( '-t', "--time", default="0")
    parser.add_argument('--delegate' , default="vx", help = 'Please Input vx or xnnpack or ethosu') 
    parser.add_argument( '-m', '--model' , default="model/mobilenetssd_facedetect_uint8_quant.tflite", help='File path of .tflite(Delect) file.') 
    parser.add_argument( '-mf', '--model_feature' , default="model/sface_quant.tflite", help='File path of .tflite(Recognition) file.') 
    parser.add_argument("--IoU", default="0.4")
    parser.add_argument("--test_img", default="img/BillGates_1.jpg")
    
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

    # 解析解譯器資訊 (人臉識別)
    interpreterFaceRecognition = InferenceDelegate(args.model_feature,args.delegate)
    interpreterFaceRecognition.allocate_tensors() 
    interpreterFaceRecognition_input_details  = interpreterFaceRecognition.get_input_details()
    interpreterFaceRecognition_output_details = interpreterFaceRecognition.get_output_details()
    iFaceRecognition_input_width    = interpreterFaceRecognition_input_details[0]['shape'][2]
    iFaceRecognition_input_height   = interpreterFaceRecognition_input_details[0]['shape'][1]
    iFaceRecognition_output_nChannel= interpreterFaceRecognition_output_details[0]['shape'][1]

    # 建立樣本比對人臉識別資訊 
    database_folder = "img/database/"
    database_path = [os.path.join(database_folder, f) for f in os.listdir(database_folder)]
    DataBase_Image = []
    DataBase_Feature  = []
    for data_path in database_path :
        img  = cv2.imread(data_path) ; DataBase_Image.append(img)
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img  = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img  = cv2.resize(img, (iFaceRecognition_input_width, iFaceRecognition_input_height))
        img  = np.expand_dims(img, axis=0).astype("float32")
        interpreterFaceRecognition.set_tensor(interpreterFaceRecognition_input_details[0]['index'], img)  # 先行進行暖開機
        interpreterFaceRecognition.invoke()
        DataBase_Feature.append( interpreterFaceRecognition.get_tensor(interpreterFaceRecognition_output_details[0]['index']) )

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
          #frame         = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          frame_resized = cv2.resize(frame, (iFaceExtractor_width, iFaceExtractor_height))

      else : 
          frame         = cv2.imread(args.test_img)
          frame         = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          frame_resized = cv2.resize(frame, (iFaceExtractor_width, iFaceExtractor_height))
    

      # 設置來源資料至解譯器、並進行推理 (人臉位置檢測)
      input_data = np.expand_dims(frame_resized, axis=0)
      interpreterFaceExtractor.set_tensor(interpreterFaceExtractor_input_details[0]['index'], input_data) 
      interpreter_time_start = time.time()
      interpreterFaceExtractor.invoke()
      interpreter_time_end   = time.time()
      if args.time =="True" or args.time == "1" :
          print( APP_NAME + " Inference Time(Face Extractor) = ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )

      # 取得解譯器的預測結果 (人臉位置檢測)
      detection_boxes   = interpreterFaceExtractor.get_tensor(interpreterFaceExtractor_output_details[0]['index'])
      detection_classes = interpreterFaceExtractor.get_tensor(interpreterFaceExtractor_output_details[1]['index'])
      detection_scores  = interpreterFaceExtractor.get_tensor(interpreterFaceExtractor_output_details[2]['index'])
      num_boxes         = interpreterFaceExtractor.get_tensor(interpreterFaceExtractor_output_details[3]['index'])

      boxs = np.squeeze(detection_boxes)
      scores = np.squeeze(detection_scores)
      boxs_nms, scores_nms = nms(boxs, scores, float(args.IoU))

      # 建立輸出結果 
      if scores_nms[0] > .5: # 當物件預測分數大於 50% 時, 取機率最大者

        # 物件位置
        x = boxs_nms[0, [1, 3]] * frame.shape[1]
        y = boxs_nms[0, [0, 2]] * frame.shape[0]
        x[0] = x[0] - 5 
        x[1] = x[1] + 5

        # --------------------------------------------------------------------------------------------------------
        #  人臉識別  : 將 人臉位置檢測器 所偵測到的人臉逐一檢測
        # --------------------------------------------------------------------------------------------------------
        #  預防邊界
        roi_x0 = max(0, np.floor(x[0] + 0.5).astype('int32'))
        roi_y0 = max(0, np.floor(y[0] + 0.5).astype('int32'))
        roi_x1 = min(frame.shape[1], np.floor(x[1] + 0.5).astype('int32'))
        roi_y1 = min(frame.shape[0], np.floor(y[1] + 0.5).astype('int32'))

        # 設置來源資料至解譯器 (人臉識別)
        if args.camera =="True" or args.camera == "1" : # 輸入端矯正
            face_img = frame[ roi_y0 : roi_y1, roi_x0 : roi_x1]
        else :
            face_img = frame[ roi_y0 : roi_y1, roi_x0 : roi_x1]

        #cv2.imwrite("WeillyLi_2.jpg",face_img)

        # 框出偵測到的物件
        cv2.rectangle(frame, ( int(x[0]), int(y[0]) ),  ( int(x[1]), int(y[1]) ), (0, 255, 0), 2) 

        # 解譯器進行設定 (人臉識別)
        face_img = cv2.resize(face_img, (iFaceRecognition_input_width, iFaceRecognition_input_height))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
        fece_input_data = np.expand_dims(face_img.astype("float32"), axis=0)
        interpreterFaceRecognition.set_tensor(interpreterFaceRecognition_input_details[0]['index'], fece_input_data) 

        # 解譯器進行推理 (人臉識別)
        interpreter_time_start = time.time()
        interpreterFaceRecognition.invoke()
        interpreter_time_end   = time.time()
        if args.time =="True" or args.time == "1" :
            print( APP_NAME + " Inference Time(Recognition) = ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )

        # 取得解譯器的預測結果 (人臉識別)
        FaceRecognitionFeature = interpreterFaceRecognition.get_tensor(interpreterFaceRecognition_output_details[0]['index'])

        # 比較相似度
        similarity = []
        for Feature in DataBase_Feature :
            similarity.append(cos_similarity(FaceRecognitionFeature[0], Feature[0]))
        similarity_max = np.max(similarity)
        similarity_idx = similarity.index(similarity_max)
        similarity_name = database_path[similarity_idx][database_path[similarity_idx].find("/")+1:database_path[similarity_idx].find("_")]
        print( "the probability of  maximum is", similarity_name ," : ", similarity_max, )

        # 建立輸出結果 (人臉識別)
        text_x = roi_x0
        text_y = min(np.floor( roi_y0-10 + 0.5 ).astype('int32'), frame.shape[0])
        if similarity_max > 0.95 :
            cv2.putText( frame, similarity_name +  " : "+ str(similarity_max)[:4], ( text_x, text_y ), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 1, cv2.LINE_AA)
        else :
            cv2.putText( frame,  "None", ( text_x, text_y ), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 1, cv2.LINE_AA)
        

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

