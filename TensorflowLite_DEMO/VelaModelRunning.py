#
# Copyright 2020-2022 NXP
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import time
import argparse
import numpy as np
import ethosu.interpreter as ethosu
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model_file',default='mobilenet_v1_0.25_224_quant_vela.tflite',help='.tflite model to be executed')
args = parser.parse_args()

interpreter = ethosu.Interpreter(args.model_file)
inputs = interpreter.get_input_details()
outputs = interpreter.get_output_details()

b, w, h, c = inputs[0]['shape']
img = np.zeros((w, h, c)).astype(inputs[0]['dtype'])
data = np.expand_dims(img, axis=0)
interpreter.set_input(0, data)

interpreter_time_n  = 20
interpreter_time_acc = 0
for i in range(interpreter_time_n):
    interpreter_time_start = time.time()
    interpreter.invoke()
    interpreter_time_end   = time.time()
    interpreter_time_acc = interpreter_time_acc + (interpreter_time_end - interpreter_time_start)
interpreter_time_acc = interpreter_time_acc/interpreter_time_n
print( "i.MX93 ethous-NPU Inference Time = ", interpreter_time_acc*1000 , " ms" )



