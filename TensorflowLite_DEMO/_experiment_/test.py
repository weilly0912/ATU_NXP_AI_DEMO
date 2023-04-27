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


import sys
import cv2
import time
import argparse
import numpy as np
from tflite_runtime.interpreter import Interpreter 
import tflite_runtime.interpreter as tflite

# --------------------------------------------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------------------------------------------
def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}

def classify_image(interpreter, top_k=1):
  output_details = interpreter.get_output_details()[0]
  output   = np.squeeze(interpreter.get_tensor(output_details['index']))
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)
  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]

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
# Main
# --------------------------------------------------------------------------------------------------------------   

# 解析解譯器資訊
interpreter = InferenceDelegate("mobilenetv2_quantized_model_pytorch.tflite","vx")
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 先行進行暖開機
if input_details[0]['dtype']==np.float32 :
   interpreter.set_tensor(input_details[0]['index'], np.zeros((1,input_details[0]['shape'][1],input_details[0]['shape'][2],input_details[0]['shape'][3])).astype("float32") )
else :
   interpreter.set_tensor(input_details[0]['index'], np.zeros((1,input_details[0]['shape'][1],input_details[0]['shape'][2],input_details[0]['shape'][3])).astype("uint8") )

interpreter.invoke()

# 解譯器進行推理
interpreter_time_start = time.time()
interpreter.invoke()
interpreter_time_end   = time.time()
print( " Inference Time = ", (interpreter_time_end - interpreter_time_start)*1000 , " ms" )

