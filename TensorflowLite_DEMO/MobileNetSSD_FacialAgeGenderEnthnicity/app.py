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
# https://www.kaggle.com/shiratorizawa/age-gender-ethnicity-classifier
# 自行隨意訓練，效果較差

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
# 標註文字
def draw_text_line(img, point, text_line: str):
    fontScale = 0.7
    thickness = 2
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    text_line = text_line.split("\n")
    baseline_acc = 0
    for i, text in enumerate(text_line):
        if text:
            text_size, baseline = cv2.getTextSize(str(text), fontFace, fontScale, thickness)
            text_loc = (point[0], point[1] + text_size[1])
            cv2.putText(img, str(text), (text_loc[0], text_loc[1] + baseline_acc), fontFace, fontScale,(0, 0, 255), thickness, 8)
            baseline_acc = baseline_acc + int(baseline*3)
    return img

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
# Define
# --------------------------------------------------------------------------------------------------------------
gender_dict = {0: "man", 1: "woman"}
ethnicity_dict = { 0:'asian', 1:'indian', 2:'black', 3:'white', 4:'middle eastern'}
facialshape_dict = { 0:'Heart', 1: 'Oblong' , 2:'Oval', 3:'Round', 4:'Square'}

# --------------------------------------------------------------------------------------------------------------
# 主程式
# --------------------------------------------------------------------------------------------------------------
def main():

    # 解析外部資訊
    APP_NAME = "FacialAgeGenderEnthnicity"
    parser = argparse.ArgumentParser()
    parser.add_argument( '-c' ,"--camera", default="0")
    parser.add_argument("--camera_format", default="V4L2_YUV2_480p")
    parser.add_argument( '-d' ,"--display", default="0")
    parser.add_argument("--save", default="1")
    parser.add_argument( '-t', "--time", default="0")
    parser.add_argument('--delegate' , default="vx", help = 'Please Input vx or xnnpack or ethosu') 
    parser.add_argument("-m","--model", default="model/mobilenetssd_facedetect_uint8_quant.tflite", help="Using facemesh_weight_flot.tflite can be accucy result")
    parser.add_argument("-mf1","--model_feature_1", default='model/facial_age_detection_quant.tflite', help="Using facemesh_weight_flot.tflite can be accucy result")
    parser.add_argument("-mf2","--model_feature_2", default='model/facial_gender_detection_quant.tflite', help="Using facemesh_weight_flot.tflite can be accucy result")
    parser.add_argument("-mf3","--model_feature_3", default='model/facial_ethnicity_detection_quant.tflite', help="Using facemesh_weight_flot.tflite can be accucy result")
    parser.add_argument("-mf4","--model_feature_4", default='model/facial_shape_detection_quant.tflite', help="Using facemesh_weight_flot.tflite can be accucy result")
    parser.add_argument("--IoU", default="0.6")
    parser.add_argument("--test_img", default="img/Didy.png")
    parser.add_argument("--offset_y", default="-20")
    
    args = parser.parse_args()
    if args.camera_format == "V4L2_YUV2_480p" : camera_format = V4L2_YUV2_480p
    if args.camera_format == "V4L2_YUV2_720p" : camera_format = V4L2_YUV2_720p
    if args.camera_format == "V4L2_H264_1080p" : camera_format = V4L2_H264_1080p

    # vela(NPU) 預設路徑修正
    if(args.delegate=="ethosu"): 
        if(args.model[-11:]!='vela.tflite') :
            args.model = args.model[:-7] + '_vela.tflite'
        if(args.model_feature_1[-11:]!='vela.tflite') :
            args.model_feature_1 = args.model_feature_1[:-7] + '_vela.tflite'
        if(args.model_feature_2[-11:]!='vela.tflite') :
            args.model_feature_2 = args.model_feature_2[:-7] + '_vela.tflite'
        if(args.model_feature_3[-11:]!='vela.tflite') :
            args.model_feature_3 = args.model_feature_3[:-7] + '_vela.tflite'      
        if(args.model_feature_4[-11:]!='vela.tflite') :
            args.model_feature_4 = args.model_feature_4[:-7] + '_vela.tflite'   

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

    # 解析解譯器資訊 (年齡)
    interpreterAge = InferenceDelegate(args.model_feature_1,args.delegate)
    interpreterAge.allocate_tensors() 
    interpreterAge_input_details  = interpreterAge.get_input_details()
    interpreterAge_output_details = interpreterAge.get_output_details()
    iAge_input_width    = interpreterAge_input_details[0]['shape'][2]
    iAge_input_height   = interpreterAge_input_details[0]['shape'][1]
    interpreterAge.set_tensor(interpreterAge_input_details[0]['index'], np.zeros((1, iAge_input_height, iAge_input_width, 1)).astype("uint8"))  # 先行進行暖開機
    interpreterAge.invoke()

    # 解析解譯器資訊 (性別)
    interpreterGender = InferenceDelegate(args.model_feature_2,args.delegate)
    interpreterGender.allocate_tensors() 
    interpreterGender_input_details  = interpreterGender.get_input_details()
    interpreterGender_output_details = interpreterGender.get_output_details()
    iGender_input_width    = interpreterGender_input_details[0]['shape'][2]
    iGender_input_height   = interpreterGender_input_details[0]['shape'][1]
    interpreterGender.set_tensor(interpreterGender_input_details[0]['index'], np.zeros((1, iGender_input_height, iGender_input_width, 1)).astype("uint8"))  # 先行進行暖開機
    interpreterGender.invoke()

    # 解析解譯器資訊 (種族)
    interpreterEthnicity = InferenceDelegate(args.model_feature_3,args.delegate)
    interpreterEthnicity.allocate_tensors() 
    interpreterEthnicity_input_details  = interpreterEthnicity.get_input_details()
    interpreterEthnicity_output_details = interpreterEthnicity.get_output_details()
    iEthnicity_input_width    = interpreterEthnicity_input_details[0]['shape'][2]
    iEthnicity_input_height   = interpreterEthnicity_input_details[0]['shape'][1]
    interpreterEthnicity.set_tensor(interpreterEthnicity_input_details[0]['index'], np.zeros((1, iEthnicity_input_height, iEthnicity_input_width, 1)).astype("uint8"))  # 先行進行暖開機
    interpreterEthnicity.invoke()

    # 解析解譯器資訊 (臉型)
    interpreterShape = InferenceDelegate(args.model_feature_4,args.delegate)
    interpreterShape.allocate_tensors() 
    interpreterShape_input_details  = interpreterShape.get_input_details()
    interpreterShape_output_details = interpreterShape.get_output_details()
    iShape_input_width    = interpreterShape_input_details[0]['shape'][2]
    iShape_input_height   = interpreterShape_input_details[0]['shape'][1]
    interpreterShape.set_tensor(interpreterShape_input_details[0]['index'], np.zeros((1, iShape_input_height, iShape_input_width, 1)).astype("uint8"))  # 先行進行暖開機
    interpreterShape.invoke()

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
      num_boxes = interpreterFaceExtractor.get_tensor(interpreterFaceExtractor_output_details[3]['index'])

      boxs = np.squeeze(detection_boxes)
      scores = np.squeeze(detection_scores)
      boxs_nms, scores_nms = nms(boxs, scores, float(args.IoU))    

      # 建立輸出結果 (每個人臉)
      for i in range( 0, len(scores_nms)-1) : 
        if scores_nms[i] > .5: # 當物件預測分數大於 50% 時
        
          # 物件位置
          x = boxs_nms[i, [1, 3]] * frame.shape[1]
          y = boxs_nms[i, [0, 2]] * frame.shape[0]
          y[1] = y[1] + int(args.offset_y)  #offset

          # 框出偵測到的物件
          cv2.rectangle(frame, ( int(x[0]), int(y[0]) ),  ( int(x[1]), int(y[1]) ), (0, 255, 0), 2)

          #  預防邊界
          roi_x0 = max(0, np.floor(x[0] + 0.5).astype('int32'))
          roi_y0 = max(0, np.floor(y[0] + 0.5).astype('int32'))
          roi_x1 = min(frame.shape[1], np.floor(x[1] + 0.5).astype('int32'))
          roi_y1 = min(frame.shape[0], np.floor(y[1] + 0.5).astype('int32'))

          # 處理來源資料

          if args.camera =="True" or args.camera == "1" : # 輸入端補正
              gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          else:
              gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)

          face_img = gray[ roi_y0 : roi_y1, roi_x0 : roi_x1]          
          face_img_resized = cv2.resize(face_img, (48, 48))
          face_input_data = np.expand_dims(face_img_resized.astype("uint8"), axis=0)
          face_input_data = np.expand_dims(face_input_data, axis=3)

          # --------------------------------------------------------------------------------------------------------
          #  檢測年齡、性別、種族、臉型
          # --------------------------------------------------------------------------------------------------------

          # 設置來源資料至解譯器 
          interpreterAge.set_tensor(interpreterAge_input_details[0]['index'], face_input_data)
          interpreterGender.set_tensor(interpreterGender_input_details[0]['index'], face_input_data) 
          interpreterEthnicity.set_tensor(interpreterEthnicity_input_details[0]['index'], face_input_data)
          interpreterShape.set_tensor(interpreterShape_input_details[0]['index'], face_input_data) 

          # 解譯器進行推理
          interpreter_time_start = time.time()
          interpreterAge.invoke()
          interpreter_time_end   = time.time()
          if args.time =="True" or args.time == "1" :
              print( "Age" + " Inference Time = ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )

          interpreter_time_start = time.time()
          interpreterGender.invoke()
          interpreter_time_end   = time.time()
          if args.time =="True" or args.time == "1" :
              print( "Gender" + " Inference Time = ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )

          interpreter_time_start = time.time()
          interpreterEthnicity.invoke()
          interpreter_time_end   = time.time()
          if args.time =="True" or args.time == "1" :
              print( "Ethnicity" + " Inference Time = ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )    

          interpreter_time_start = time.time()
          interpreterShape.invoke()
          interpreter_time_end   = time.time()
          if args.time =="True" or args.time == "1" :
              print( "Face Shape" + " Inference Time = ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )   


          # 取得解譯器的預測結果 
          AgePredict       = interpreterAge.get_tensor(interpreterAge_output_details[0]['index'])
          GenderPredict    = interpreterGender.get_tensor(interpreterGender_output_details[0]['index'])
          EthnicityPredict = interpreterEthnicity.get_tensor(interpreterEthnicity_output_details[0]['index'])
          ShapePredict     = interpreterShape.get_tensor(interpreterShape_output_details[0]['index'])

          # 建立輸出結果 (面網檢測)
          Age       = AgePredict[0]
          gender    = gender_dict[np.argmax(GenderPredict)]
          ethnicity = ethnicity_dict[np.argmax(EthnicityPredict)]
          facialshape = facialshape_dict[np.argmax(ShapePredict)]
          text_info = "Age : " +  str(Age) + " \n"  + "Gender : " + gender + "\n" + "Ethnicity : " + ethnicity + "\n" + "Shpae : " + facialshape 
          print(text_info)

          #emotion text
          text_x = roi_x0
          text_y = min(np.floor( roi_y0 + 0.5 ).astype('int32'), frame.shape[0])
          frame  = draw_text_line(frame,(text_x, text_y), text_info)

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

