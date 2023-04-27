# WPI Confidential Proprietary
#--------------------------------------------------------------------------------------
# Copyright (c) 2021 Freescale Semiconductor
# Copyright 2021 WPI
# All Rights Reserved
##--------------------------------------------------------------------------------------
# * Code Ver : 1.0
# * Code Date: 2023/02/20
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
# https://github.com/ultralytics/yolov5
# https://github.com/tangsanli5201/DeepPCB.git


import sys
import cv2
import torch
import torchvision
import time
import argparse
import numpy as np
import tflite_runtime.interpreter as tflite
from plots import Annotator, colors

# --------------------------------------------------------------------------------------------------------------
# 主程式
# --------------------------------------------------------------------------------------------------------------
def main():

    # 解析外部資訊
    APP_NAME = "YOLOv8n-ObejctDetect"
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", default="0")
    args = parser.parse_args()

    # 是否啟用攝鏡頭
    cap = cv2.VideoCapture("v4l2src device=/dev/video3 ! video/x-raw,format=YUY2,width=640,height=480,framerate=30/1! imxvideoconvert_g2d ! video/x-raw,format=RGB16 ! videoconvert ! appsink")
    if(cap.isOpened()==False) :
        print( "Open Camera Failure !!")
        sys.exit()
    else :
        print( "Open Camera Success !!")

    # 迴圈 / 重複推理       
    while(True):
      
      # 視訊/影像資料來源
      time_start = time.time()
      ret, frame    = cap.read()
      time_end = time.time()
      print(" Time spend= ", (time_end - time_start)*1000 , " ms")

      # 顯示輸出結果
      cv2.imshow('frame', frame.astype('uint8'))
      if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()