# WPI Confidential Proprietary
#--------------------------------------------------------------------------------------
# Copyright (c) 2020 Freescale Semiconductor
# Copyright 2020 WPI
# All Rights Reserved
##--------------------------------------------------------------------------------------
# * Code Ver : 1.0
# * Code Date: 2021/7/30
# * Author   : Weilly Li
# * Non-qunt Model*
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
# https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/raspberry_pi

import sys
import cv2
import time
import argparse
import numpy as np
from scipy.io import wavfile
import tflite_runtime.interpreter as tflite

# --------------------------------------------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------------------------------------------
def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}

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
# 主程式
# --------------------------------------------------------------------------------------------------------------
def main():

    # 取得外部輸入資訊
    APP_NAME = "BirdsClassification"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = argparse.ArgumentParser()
    parser.add_argument( '-d' ,"--display", default="0")
    parser.add_argument("--save", default="0")
    parser.add_argument( '-t', "--time", default="0")
    parser.add_argument('--delegate' , default="vx", help = 'Please Input vx or xnnpack or ethosu') 
    parser.add_argument( '-m', '--model'   , default="model/brids-classification.tflite", help='File path of .tflite file.')
    parser.add_argument('--labels'  , default="label/label.txt", help='File path of labels file.')
    parser.add_argument('--test_audio', default="audio/XC563091.wav", help='File path of labels file.')
    args = parser.parse_args()
    print("TFLite doesn't support complex and can't generate spectrogram using it.")

    # vela(NPU) 預設路徑修正
    if(args.delegate=="ethosu"): 
        if(args.model[-11:]!='vela.tflite') :
            args.model = args.model[:-7] + '_vela.tflite'

    # 載入標籤
    labels = load_labels(args.labels)
    test_data_label = ['azaspi1', 'chcant2', 'houspa', 'redcro', 'wbwwre1']

    # 解析解譯器資訊
    interpreter = InferenceDelegate(args.model,args.delegate)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    sample_rate  = input_details[0]["shape"][1]

    # 先行進行暖開機
    interpreter.set_tensor(input_details[0]['index'], np.zeros((1,sample_rate)).astype("float32") )
    interpreter.invoke()

    # 載入音檔 -> 並以取樣頻率分段
    sample_rate_, audio_data = wavfile.read(args.test_audio, 'rb')
    n = int(audio_data.shape[0] / sample_rate) + 1
    audio_data = np.array(audio_data) / (32767)
    audio_data_array = np.zeros(n*sample_rate)
    audio_data_array[:len(audio_data)] = audio_data
    splitted_audio_data = np.reshape(audio_data_array, (n, sample_rate))

    # 分析與推理
    results = []
    interpreter_time_accumulation = 0
    for i, data in enumerate(splitted_audio_data):
        
        # 設定分段後的數據資料
        input_data = np.expand_dims(data, axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data.astype("float32"))

        # 進行推理
        interpreter_time_start = time.time()
        interpreter.invoke()
        interpreter_time_end   = time.time()
        interpreter_time_accumulation = interpreter_time_accumulation + (interpreter_time_end - interpreter_time_start)
        
        # 取得輸出資訊
        yamnet_output = interpreter.get_tensor(output_details[0]['index'])
        inference = interpreter.get_tensor(output_details[1]['index'])
        
        # 解析輸出
        results.append(inference[0])
        result_index = np.argmax(inference[0])
        spec_result_index = np.argmax(yamnet_output[0])
        t = labels[spec_result_index]

        # 印出資訊
        result_str = f'Result of the window {i}: ' \
        f'\t{test_data_label[result_index]} -> {inference[0][result_index]:.3f}, ' \
        f'\t({labels[spec_result_index]} -> {yamnet_output[0][spec_result_index]:.3f})'
        print(result_str)

    results_np = np.array(results)
    mean_results = results_np.mean(axis=0)
    result_index = mean_results.argmax()
    print(f'Mean result: {test_data_label[result_index]} -> {mean_results[result_index]}')
    print( "Note : this demo isn't use NPU.")

    if args.time =="True" or args.time == "1" :
        print( APP_NAME + " Inference Time = ", (interpreter_time_accumulation)*1000 , " ms" )
        
if __name__ == "__main__":
    main()
 