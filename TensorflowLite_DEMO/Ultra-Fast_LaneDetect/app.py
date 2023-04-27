# WPI Confidential Proprietary
#--------------------------------------------------------------------------------------
# Copyright (c) 2020 Freescale Semiconductor
# Copyright 2020 WPI
# All Rights Reserved
##--------------------------------------------------------------------------------------
# * Code Ver : 3.0
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
lane_colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]

tusimple_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
			116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
			168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
			220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
			272, 276, 280, 284]

mean=[0.485, 0.456, 0.406]
std =[0.229, 0.224, 0.225]
class ModelConfig():
	def __init__(self):
		self.img_w = 1280
		self.img_h = 720
		self.row_anchor = tusimple_row_anchor
		self.griding_num = 100
		self.cls_num_per_lane = 56

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def process_output(output, cfg):		
  # Parse the output of the model to get the lane informatio
  processed_output = output[:, ::-1, :]
  prob = softmax(processed_output[:-1, :, :])
  idx = np.arange(cfg.griding_num) + 1
  idx = idx.reshape(-1, 1, 1)
  loc = np.sum(prob * idx, axis=0)
  processed_output = np.argmax(processed_output, axis=0)
  loc[processed_output == cfg.griding_num] = 0
  processed_output = loc

  col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
  col_sample_w = col_sample[1] - col_sample[0]

  lane_points_mat = []
  lanes_detected = []

  max_lanes = processed_output.shape[1]
  for lane_num in range(max_lanes):
    lane_points = []
    # Check if there are any points detected in the lane
    if np.sum(processed_output[:, lane_num] != 0) > 2:
      lanes_detected.append(True)

      # Process each of the points for each lane
      for point_num in range(processed_output.shape[0]):
        if processed_output[point_num, lane_num] > 0:
          lane_point = [int(processed_output[point_num, lane_num] * col_sample_w * cfg.img_w / 800) - 1, int(cfg.img_h * (cfg.row_anchor[cfg.cls_num_per_lane-1-point_num]/288)) - 1 ]
          lane_points.append(lane_point)
    else:
      lanes_detected.append(False)

    lane_points_mat.append(lane_points)
  return np.array(lane_points_mat), np.array(lanes_detected)

def draw_lanes(input_img, lane_points_mat, lanes_detected, cfg, draw_points=True):
  # Write the detected line points in the image
  visualization_img = cv2.resize(input_img, (cfg.img_w, cfg.img_h), interpolation = cv2.INTER_AREA)

  # Draw a mask for the current lane
  if(lanes_detected[1] and lanes_detected[2]):
    lane_segment_img = visualization_img.copy()
    cv2.fillPoly(lane_segment_img, pts = [np.vstack((lane_points_mat[1],np.flipud(lane_points_mat[2])))], color =(255,191,0))
    visualization_img = cv2.addWeighted(visualization_img, 0.7, lane_segment_img, 0.3, 0)
  
  if(draw_points):
    for lane_num,lane_points in enumerate(lane_points_mat):
      for lane_point in lane_points:
        cv2.circle(visualization_img, (lane_point[0],lane_point[1]), 3, lane_colors[lane_num], -1)

  return visualization_img

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

cfg = ModelConfig()

# --------------------------------------------------------------------------------------------------------------
# 主程式
# --------------------------------------------------------------------------------------------------------------
def main():

    # 解析外部資訊
    APP_NAME = "laneDetector"
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="0")
    parser.add_argument("--camera_format", default="V4L2_YUV2_480p")
    parser.add_argument("--display", default="0")
    parser.add_argument("--save", default="1")
    parser.add_argument("--time", default="0")
    parser.add_argument('--delegate' , default="vx", help = 'Please Input nnapi or xnnpack')
    parser.add_argument('--model' , default="LDW_tusimple_integer_quant.tflite", help='File path of .tflite file.')
    parser.add_argument("--IoU", default="0.6")
    parser.add_argument("--test_img", default="lane_detection.jpg")
    
    args = parser.parse_args()
    if args.camera_format == "V4L2_YUV2_480p" : camera_format = V4L2_YUV2_480p
    if args.camera_format == "V4L2_YUV2_720p" : camera_format = V4L2_YUV2_720p
    if args.camera_format == "V4L2_H264_1080p" : camera_format = V4L2_H264_1080p

    # 解析解譯器資訊
    interpreter = InferenceDelegate(args.model,args.delegate)
    interpreter.allocate_tensors() 
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    width    = input_details[0]['shape'][2]
    height   = input_details[0]['shape'][1]
    nChannel = input_details[0]['shape'][3]
    num_anchors = output_details[0]['shape'][1]
    num_lanes = output_details[0]['shape'][2]	
    num_points = output_details[0]['shape'][3]
    
    # 先行進行暖開機
    interpreter.set_tensor(input_details[0]['index'], np.zeros((1,height,width,nChannel)).astype("float32") )
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
          ret, frame    = cap.read()
          frame_resized = cv2.resize(frame, (width, height))

      else : 
          frame         = cv2.imread(args.test_img)
          frame_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          frame_resized = cv2.resize(frame_rgb, (width, height))
    
      # 設置來源資料至解譯器
      input_data = ((frame_resized/ 255.0 - mean) / std).astype(np.float32)
      input_data = input_data[np.newaxis,:,:,:] 
      interpreter.set_tensor(input_details[0]['index'], input_data) 

      # 解譯器進行推理
      interpreter_time_start = time.time()
      interpreter.invoke()
      interpreter_time_end   = time.time()
      if args.time =="True" or args.time == "1" :
          print( APP_NAME + " Inference Time = ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )

      # 取得解譯器的預測結果
      output = interpreter.get_tensor(output_details[0]['index']).reshape(num_anchors, num_lanes, num_points)
      lanes_points, lanes_detected = process_output(output, cfg)
      frame = draw_lanes(frame, lanes_points, lanes_detected, cfg, draw_points=True)

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