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
# https://github.com/tantara/JejuNet
# https://github.com/tensorflow/models/tree/master/research/deeplab
#
# Runtime :
# mobilenet_v2_deeplab_v3_256_quant => 42 ms

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
def get_classes(classes_path): # 定義類別的解析函示
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

class_names = get_classes("label/voc_classes.txt")

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
    APP_NAME = "SegmationObjection"
    parser = argparse.ArgumentParser()
    parser.add_argument( '-c' ,"--camera", default="0")
    parser.add_argument("--camera_format", default="V4L2_YUV2_480p")
    parser.add_argument( '-d' ,"--display", default="0")
    parser.add_argument("--save", default="1")
    parser.add_argument( '-t', "--time", default="0")
    parser.add_argument('--delegate' , default="vx", help = 'Please Input vx or xnnpack or ethosu') 
    parser.add_argument( '-m', '--model' , default="model/mobilenet_v2_deeplab_v3_256_quant.tflite", help='File path of .tflite file.')
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
    interpreter.set_tensor(input_details[0]['index'], np.zeros((1,height,width,nChannel)).astype("uint8") )
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
        input_data = np.expand_dims(frame_resized.astype("uint8"), axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data) 

        # 解譯器進行推理
        interpreter_time_start = time.time()
        interpreter.invoke()
        interpreter_time_end   = time.time()
        if args.time =="True" or args.time == "1" :
            print( APP_NAME + " Inference Time = ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )

        # 取得解譯器的預測結果
        prediction    = interpreter.get_tensor(output_details[0]['index']) 
        prediction_img= np.reshape(prediction, [width, height]) 

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
        
        if args.camera =="False" or args.camera == "0" :
            frame_resized= cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB) # 校正顏色
        
        image_result =  cv2.add( frame_resized.astype('uint8'), image_rgb_merge.astype('uint8') )
        image_result =  cv2.resize(image_result, (frame.shape[1], frame.shape[0]))
        print("Segmation Result : Class of Num = ", labelcount_len , ' , ' ,\
               "class of MAX = ", class_names[np.where(labelbinCnt==labelcount[labelcount_len-2])[0][0] - 1])

        
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