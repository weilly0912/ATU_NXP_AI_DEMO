
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
# References: NXP

import sys
import cv2
import time
import argparse
import numpy as np
import tflite_runtime.interpreter as tflite

# --------------------------------------------------------------------------------------------------------------
# Define
# --------------------------------------------------------------------------------------------------------------
# Person
#set cam param
eye_height = 200
eye_angle = 60
eye_slope = 60

# Mask
variances = [0.1, 0.2]
priors = np.reshape(np.loadtxt('priors.txt'), (9949, 4))
labels_facemask =  {0: 'Mask', 1: 'NoMask'}

# CAMERA
V4L2_YUV2_480p = "v4l2src device=/dev/video3 ! video/x-raw,format=YUY2,width=640,height=480, pixel-aspect-ratio=1/1, framerate=30/1! videoscale!videoconvert ! appsink" 
V4L2_YUV2_720p = "v4l2src device=/dev/video3 ! video/x-raw,format=YUY2,width=1280,height=720, pixel-aspect-ratio=1/1, framerate=30/1! videoscale!videoconvert ! appsink"                           
V4L2_H264_1080p = "v4l2src device=/dev/video3 ! video/x-h264, width=1920, height=1080, pixel-aspect-ratio=1/1, framerate=30/1 ! queue ! h264parse ! vpudec ! queue ! queue leaky=1 ! videoscale ! videoconvert ! appsink"

# --------------------------------------------------------------------------------------------------------------
# API For NXP
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
    
def pad_input_image(img):
    """pad image to suitable shape"""
    img_h, img_w, _ = img.shape

    img_pad_h = 0
    img_pad_w = 0

    if img_w > img_h:
        img_pad_h = img_w - img_h
    else:
        img_pad_w = img_h - img_w

    img = cv2.copyMakeBorder(img, 0, img_pad_h, 0, img_pad_w,cv2.BORDER_CONSTANT)
    return img

def single_class_non_max_suppression(bboxes, confidences, conf_thresh=0.6, iou_thresh=0.5, keep_top_k=-1):
    '''
    do nms on single class.
    Hint: for the specific class, given the bbox and its confidence,
    1) sort the bbox according to the confidence from top to down, we call this a set
    2) select the bbox with the highest confidence, remove it from set, and do IOU calculate with the rest bbox
    3) remove the bbox whose IOU is higher than the iou_thresh from the set,
    4) loop step 2 and 3, util the set is empty.
    :param bboxes: numpy array of 2D, [num_bboxes, 4]
    :param confidences: numpy array of 1D. [num_bboxes]
    :param conf_thresh:
    :param iou_thresh:
    :param keep_top_k:
    :return:
    '''

    if len(bboxes) == 0: return []

    conf_keep_idx = np.where(confidences > conf_thresh)[0]
    bboxes = bboxes[conf_keep_idx]
    confidences = confidences[conf_keep_idx]

    pick = []
    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    xmax = bboxes[:, 2]
    ymax = bboxes[:, 3]

    area = (xmax - xmin + 1e-3) * (ymax - ymin + 1e-3)
    idxs = np.argsort(confidences)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # keep top k
        if keep_top_k != -1:
            if len(pick) >= keep_top_k:
                break

        overlap_xmin = np.maximum(xmin[i], xmin[idxs[:last]])
        overlap_ymin = np.maximum(ymin[i], ymin[idxs[:last]])
        overlap_xmax = np.minimum(xmax[i], xmax[idxs[:last]])
        overlap_ymax = np.minimum(ymax[i], ymax[idxs[:last]])
        overlap_w = np.maximum(0, overlap_xmax - overlap_xmin)
        overlap_h = np.maximum(0, overlap_ymax - overlap_ymin)
        overlap_area = overlap_w * overlap_h
        overlap_ratio = overlap_area / (area[idxs[:last]] + area[i] - overlap_area)

        need_to_be_deleted_idx = np.concatenate(([last], np.where(overlap_ratio > iou_thresh)[0]))
        idxs = np.delete(idxs, need_to_be_deleted_idx)

    return conf_keep_idx[pick]

def decode_bbox(bbox, priors, variances):
    """
    Decode locations from predictions using anchors to undo
    the encoding we did for offset regression at train time.
    """
    if variances is None:
        variances = [0.1, 0.2]

    boxes = np.concatenate((priors[:, :2] + bbox[:, :2] * variances[0] * priors[:, 2:],priors[:, 2:] * np.exp(bbox[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def softmax(x):
    orig_shape = x.shape
    if len(x.shape) > 1:
        # matrix
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
        #print("matrix")
    else:
        # vector
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
        #print("vector")
    return x 

# --------------------------------------------------------------------------------------------------------------
# 主程式
# --------------------------------------------------------------------------------------------------------------
def main():

    # 解析外部資訊
    APP_NAME = "MaskDetector"
    parser = argparse.ArgumentParser()
    parser.add_argument( '-c' ,"--camera", default="0")
    parser.add_argument("--camera_format", default="V4L2_YUV2_480p")
    parser.add_argument( '-d' ,"--display", default="0")
    parser.add_argument("--save", default="1")
    parser.add_argument( '-t', "--time", default="0")
    parser.add_argument('--delegate' , default="ethosu", help = 'Please Input vx or xnnpack or ethosu') 
    parser.add_argument('-m',"--model", default="mobilenet_ssd_v2_coco_quant_postprocess.tflite")
    parser.add_argument('-mf',"--model_feature", default="facemask_int8.tflite")
    parser.add_argument("--IoU", default="0.6")
    parser.add_argument("--test_img", default="crowd.jpg")
    parser.add_argument("--social_distance", default="1")
    parser.add_argument("--mask_detector", default="1")
    parser.add_argument("--fontsize", default="1.5")
    
    args = parser.parse_args()
    if args.camera_format == "V4L2_YUV2_480p" : camera_format = V4L2_YUV2_480p
    if args.camera_format == "V4L2_YUV2_720p" : camera_format = V4L2_YUV2_720p
    if args.camera_format == "V4L2_H264_1080p" : camera_format = V4L2_H264_1080p

    # vela(NPU) 路徑修正
    if(args.delegate=="ethosu"): args.model = 'output/' + args.model[:-7] + '_vela.tflite'

    # 解析解譯器資訊 (人臉位置檢測)
    interpreterPersonExtractor = InferenceDelegate(args.model,args.delegate)
    interpreterPersonExtractor.allocate_tensors() 
    interpreterPersonExtractor_input_details  = interpreterPersonExtractor.get_input_details()
    interpreterPersonExtractor_output_details = interpreterPersonExtractor.get_output_details()
    iPersonExtractor_width    = interpreterPersonExtractor_input_details[0]['shape'][2]
    iPersonExtractor_height   = interpreterPersonExtractor_input_details[0]['shape'][1]
    iPersonExtractor_nChannel = interpreterPersonExtractor_input_details[0]['shape'][3]
    interpreterPersonExtractor.set_tensor(interpreterPersonExtractor_input_details[0]['index'], np.zeros((1,iPersonExtractor_height,iPersonExtractor_width,iPersonExtractor_nChannel)).astype("uint8") ) # 先行進行暖開機
    interpreterPersonExtractor.invoke()

    # 解析解譯器資訊 (面網檢測)
    interpreterMaskDetector = InferenceDelegate(args.model_feature,args.delegate)
    interpreterMaskDetector.allocate_tensors() 
    interpreterMaskDetector_input_details  = interpreterMaskDetector.get_input_details()
    interpreterMaskDetector_output_details = interpreterMaskDetector.get_output_details()
    iMaskDetector_input_width    = interpreterMaskDetector_input_details[0]['shape'][2]
    iMaskDetector_input_height   = interpreterMaskDetector_input_details[0]['shape'][1]
    iMaskDetector_input_nChannel= interpreterMaskDetector_input_details[0]['shape'][3]
    interpreterMaskDetector.set_tensor(interpreterMaskDetector_input_details[0]['index'], np.zeros((1,iMaskDetector_input_height,iMaskDetector_input_width,iMaskDetector_input_nChannel)).astype("float32"))  # 先行進行暖開機
    interpreterMaskDetector.invoke()

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
      else : 
          frame = cv2.imread(args.test_img)

      result = frame.copy()

      # --------------------------------------------------------------------------------------------------------
      #  社交距離偵測
      # --------------------------------------------------------------------------------------------------------
      if args.social_distance =="True" or args.social_distance == "1" :
          
          # 設置來源資料至解譯器、並進行推理 (行人位置檢測)
          if args.camera =="True" or args.camera == "1" :
              frame_resized = cv2.resize(frame, (iPersonExtractor_width, iPersonExtractor_height))
          else : 
              frame_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
              frame_resized = cv2.resize(frame_rgb, (iPersonExtractor_width, iPersonExtractor_height))
          
          input_data = np.expand_dims(frame_resized, axis=0)
          interpreterPersonExtractor.set_tensor(interpreterPersonExtractor_input_details[0]['index'], input_data) 

          # 解譯器進行推理 (口罩位置檢測)
          interpreter_time_start = time.time()
          interpreterPersonExtractor.invoke()
          interpreter_time_end   = time.time()
          if args.time =="True" or args.time == "1" :
              print( APP_NAME + " Inference Time (Person Extractor) = ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )  

          # 取得解譯器的預測結果 (行人位置檢測)
          detection_boxes   = interpreterPersonExtractor.get_tensor(interpreterPersonExtractor_output_details[0]['index'])
          detection_classes = interpreterPersonExtractor.get_tensor(interpreterPersonExtractor_output_details[1]['index'])
          detection_scores  = interpreterPersonExtractor.get_tensor(interpreterPersonExtractor_output_details[2]['index'])
          num_boxes         = interpreterPersonExtractor.get_tensor(interpreterPersonExtractor_output_details[3]['index'])
        
          boxs    = np.squeeze(detection_boxes)
          scores  = np.squeeze(detection_scores)
          classes = np.squeeze(detection_classes+1).astype(np.int32)
          
          # 建立輸出結果 (定義人的位置)
          person_location = []
          for i in range( 0, len(scores)-1) : 
              if scores[i] > .5 and classes[i] == 1 :  # 當物件預測分數大於 50% 時  
                # 物件位置
                x = boxs[i, [1, 3]] * frame.shape[1]
                y = boxs[i, [0, 2]] * frame.shape[0]
                
                top = max(0, np.floor(y[0] + 0.5).astype('int32'))
                left = max(0, np.floor(x[0] + 0.5).astype('int32'))
                bottom = min(frame.shape[0], np.floor(y[1] + 0.5).astype('int32'))
                right  = min(frame.shape[1], np.floor(x[1] + 0.5).astype('int32'))
                
                #method 1, not calculate > 90° range
                angle_y =(1 - bottom / frame.shape[0]) * eye_angle + (eye_slope - eye_angle / 2)
                if (angle_y >= 90):
                    continue
                angle_x = ((left + right) / 2 / frame.shape[1] - 0.5) * (eye_angle / 2)
                Y = eye_height * np.tan(angle_y * np.pi / 180)
                X = Y * np.tan(angle_x * np.pi / 180) * 1.4
                person_location.append({'top':top, 'left': left, 'bottom':bottom,'right':right, 'X':X, 'Y':Y, 'warning': False})

        
          # 建立輸出結果 (相互比較社交距離)
          persons = len(person_location)
          for index_1 in range(persons-1):
              for j  in range(persons - index_1 - 1):
                  index_2 = index_1 + j + 1
                  if abs(person_location[index_2]['X'] - person_location[index_1]['X']) < 100 and abs(person_location[index_2]['Y'] - person_location[index_1]['Y']) < 100:
                      rect1_x = (int)((person_location[index_1]['right'] +person_location[index_1]['left']) / 2)
                      rect1_y = (int)((person_location[index_1]['top'] + person_location[index_1]['bottom']) / 2)
                      rect2_x = (int)((person_location[index_2]['right'] + person_location[index_2]['left']) / 2)
                      rect2_y = (int)((person_location[index_2]['top'] + person_location[index_2]['bottom']) / 2)
                      person_location[index_1]['warning'] = True
                      person_location[index_2]['warning'] = True
                      #cv2.line(frame,(rect1_x,rect1_y),(rect2_x,rect2_y),(0,0,255), 3)

          # 建立輸出結果 (畫出訊息)
          for index in range(persons):
              if person_location[index]['warning'] == True:
                  color = (0, 40, 200)
              else:
                  color = (128, 128, 128)
                  
              left = (int)(person_location[index]['left'])
              top  = (int)(person_location[index]['top'])
              right = (int)(person_location[index]['right'])
              bottom  = (int)(person_location[index]['bottom'])
              
              mask = np.zeros((result.shape),dtype=np.uint8)
              cv2.rectangle(mask, (left, top), (right, bottom), color, -1)
              result = 0.4*mask + result
              result[result>=255] = 255

      # --------------------------------------------------------------------------------------------------------
      #  口罩偵測
      # --------------------------------------------------------------------------------------------------------  
      if args.mask_detector =="True" or args.mask_detector == "1" :

        # 設置來源資料至解譯器、並進行推理 (口罩位置檢測)
        frame_pad     = pad_input_image(frame.astype("float32"))
        frame_resized = cv2.resize(frame_pad, (iMaskDetector_input_width, iMaskDetector_input_height)) 
        frame_resized = frame_resized / 255.0 -0.5
        frame_resized = frame_resized/0.5 #by weilly
        if args.camera =="False" or args.camera == "0" : frame_resized  = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        iMaskDetector_input_data = np.expand_dims(frame_resized, axis=0)
        interpreterMaskDetector.set_tensor(interpreterMaskDetector_input_details[0]['index'], iMaskDetector_input_data) 
        
        # 解譯器進行推理 (口罩位置檢測)
        interpreter_time_start = time.time()
        interpreterMaskDetector.invoke()
        interpreter_time_end   = time.time()
        if args.time =="True" or args.time == "1" :
            print( APP_NAME + " Inference Time (Mask Detector)= ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )  

        # 取得解譯器的預測結果 (面網檢測)
        predictions = interpreterMaskDetector.get_tensor(interpreterMaskDetector_output_details[0]['index'])
        bbox_regressions, confs = np.split(predictions[0], [4, ], axis=-1)
        boxes = decode_bbox(bbox_regressions, priors, variances)
        confs = softmax(confs)
        confs = confs[:, [1, 2]]
        bbox_max_scores = np.max(confs, axis=1)
        bbox_max_score_classes = np.argmax(confs, axis=1)   
        keep_idxs = single_class_non_max_suppression(boxes,bbox_max_scores,conf_thresh=0.5,iou_thresh=0.5,keep_top_k=-1)
        
        # 畫出結果
        height_pad, width_pad, _ = frame_pad.shape
        for idx in keep_idxs:
            conf = float(bbox_max_scores[idx])
            class_id = bbox_max_score_classes[idx]
            bbox = boxes[idx]
            # clip the coordinate, avoid the value exceed the image boundary.
            xmin = max(0, int(bbox[0] * width_pad))
            ymin = max(0, int(bbox[1] * height_pad))
            xmax = min(int(bbox[2] * width_pad), frame.shape[1])
            ymax = min(int(bbox[3] * height_pad), frame.shape[0])
            if class_id== 0 :
                color = (0, 255, 0)
            else :
                color = (0, 0, 255)

            cv2.rectangle(result, ( xmin, ymin ),  ( xmax, ymax ), color, 2) 
            cv2.putText(result, str(labels_facemask[class_id]), (xmin + 2, ymin - 2),cv2.FONT_HERSHEY_COMPLEX_SMALL, float(args.fontsize), color)
      
      # 顯示輸出結果
      if args.save == "True" or args.save == "1" :
          cv2.imwrite( APP_NAME + "-" + args.test_img[:len(args.test_img)-4] +'_result.jpg', result.astype("uint8"))
          print("Save Reuslt Image Success , " + APP_NAME + "-" +  args.test_img[:len(args.test_img)-4] + '_result.jpg')

      if args.display =="True" or args.display == "1" :
          cv2.imshow('frame', result.astype('uint8'))
          if cv2.waitKey(1) & 0xFF == ord('q'): break

      if (args.display =="False" or args.display == "0") and( args.camera =="False" or args.camera == "0" ) : sys.exit()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


