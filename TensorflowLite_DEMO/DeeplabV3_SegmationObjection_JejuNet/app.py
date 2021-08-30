# WPI Confidential Proprietary
#--------------------------------------------------------------------------------------
# Copyright (c) 2020 Freescale Semiconductor
# Copyright 2020 WPI
# All Rights Reserved
##--------------------------------------------------------------------------------------
# * Code Ver : 1.0
# * Code Date: 2021/7/30
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
# https://github.com/tantara/JejuNet
# https://github.com/tensorflow/models/tree/master/research/deeplab

import sys
import cv2
import time
import argparse
import numpy as np
from tflite_runtime.interpreter import Interpreter 

# --------------------------------------------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------------------------------------------
def get_classes(classes_path): # 定義類別的解析函示
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

class_names = get_classes("voc_classes.txt")

# --------------------------------------------------------------------------------------------------------------
# 主程式
# --------------------------------------------------------------------------------------------------------------
def main():

    # 解析外部資訊
    APP_NAME = "SegmationObjection"
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="0")
    parser.add_argument("--display", default="0")
    parser.add_argument("--save", default="1")
    parser.add_argument("--time", default="0")
    parser.add_argument("--test_img", default="dog416.png")
    args = parser.parse_args()

    # 解析解譯器資訊
    interpreterSegmetation = Interpreter(model_path='mobilenet_v2_deeplab_v3_256_quant.tflite')
    interpreterSegmetation.allocate_tensors() 
    input_details = interpreterSegmetation.get_input_details()
    output_details = interpreterSegmetation.get_output_details()
    width = input_details[0]['shape'][2]
    height = input_details[0]['shape'][1]
    nChannel = input_details[0]['shape'][3]

    # 先行進行暖開機
    interpreterSegmetation.set_tensor(input_details[0]['index'], np.zeros((1,height,width,nChannel)).astype("uint8") )
    interpreterSegmetation.invoke()

    # 是否啟用攝鏡頭
    if args.camera =="True" or args.camera == "1" :
        cap = cv2.VideoCapture("v4l2src device=/dev/video1 ! video/x-raw,format=YUY2,width=1280,height=720,framerate=30/1! videoscale!videoconvert ! appsink")
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
        input_data = np.expand_dims(frame_resized.astype("uint8"), axis=0)
        interpreterSegmetation.set_tensor(input_details[0]['index'], input_data) 

        # 解譯器進行推理
        interpreter_time_start = time.time()
        interpreterSegmetation.invoke()
        interpreter_time_end   = time.time()
        if args.time =="True" or args.time == "1" :
            print( APP_NAME + " Inference Time = ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )

        # 取得解譯器的預測結果
        prediction  = interpreterSegmetation.get_tensor(output_details[0]['index']) 
        prediction_img  = np.reshape(prediction, [width, height])

        # 以機率分布重新排列標籤, 由小到大排列成序列資料
        labelIndex = np.argmax(prediction[0])
        labelbinCnt= np.bincount(prediction[0])
        labelcount = np.unique(np.sort(labelbinCnt)) 
        labelcount_len = len(labelcount)
        #labelSort  = np.unique(np.sort(prediction[0]))# print(labelSort)

        # 建立輸出結果 (待重新設計)
        image_rgb_merge = np.zeros((width,height,3), dtype=np.float32)
        if(labelcount_len==3):
            image_rgb_merge[prediction_img==np.where(labelbinCnt==labelcount[labelcount_len-2])[0]] = (0,200,200)

        if(labelcount_len==4):
            image_rgb_merge[prediction_img==np.where(labelbinCnt==labelcount[labelcount_len-2])[0]] = (0,200,200)
            image_rgb_merge[prediction_img==np.where(labelbinCnt==labelcount[labelcount_len-3])[0]] = (0,0,200)

        if(labelcount_len>4):
            image_rgb_merge[prediction_img==np.where(labelbinCnt==labelcount[labelcount_len-2])[0]] = (0,200,200)
            image_rgb_merge[prediction_img==np.where(labelbinCnt==labelcount[labelcount_len-3])[0]] = (0,0,200)
            image_rgb_merge[prediction_img==np.where(labelbinCnt==labelcount[labelcount_len-4])[0]] = (200,0,200)
        frame_resized= cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB) # 校正顏色
        image_result =  cv2.add( frame_resized.astype('uint8'), image_rgb_merge.astype('uint8') )
        image_result =  cv2.resize(image_result, (frame.shape[1], frame.shape[0]))
        print("Segmation Result : Class of Num = ", labelcount_len , ' , ' ,\
               "class of MAX = ", class_names[np.where(labelbinCnt==labelcount[labelcount_len-2])[0][0] - 1])

        
        # 顯示輸出結果
        if args.save == "True" or args.save == "1" :
            cv2.imwrite( APP_NAME + "-" + args.test_img[:len(args.test_img)-4] +'_result.jpg', image_result.astype("uint8"))
            print("Save Reuslt Image Success , " + APP_NAME + '_result.jpg')

        if args.display =="True" or args.display == "1" :
            cv2.imshow('frame', image_result.astype('uint8'))
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        if args.camera =="False" or args.camera == "0" : sys.exit()
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()