from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter


def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]


def main():
  #input 
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--model', help='File path of .tflite file.', required=True)
  parser.add_argument('--labels', help='File path of labels file.', required=True)
  parser.add_argument('--image', help='File path of labels file.', required=True)
  args = parser.parse_args()
  labels = load_labels(args.labels)

  #tensor setting
  interpreter = Interpreter(args.model)
  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']

  #load image
  frame         = cv2.imread(args.image)
  frame_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Acquire frame and resize to expected shape [1xHxWx3]
  frame_resized = cv2.resize(frame_rgb, (width, height))

  #start inference
  start_time = time.time()
  results = classify_image(interpreter, frame_resized)
  elapsed_ms = (time.time() - start_time) * 1000

  print("最大分類", results)

if __name__ == "__main__":
    main()
 