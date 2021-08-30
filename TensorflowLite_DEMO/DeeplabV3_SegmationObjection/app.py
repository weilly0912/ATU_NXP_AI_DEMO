import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter 

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def create_pascal_label_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    index = np.arange(256, dtype=int)
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((index >> channel) & 1) << shift
        index >>= 3
    return colormap

#class_names = get_classes("voc_classes.txt")

# --------------------------------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------------------------------
interpreterSegmetation = Interpreter(model_path='deeplabv3_tf22_weight_uint8.tflite')

interpreterSegmetation.allocate_tensors() 
input_details = interpreterSegmetation.get_input_details()
output_details = interpreterSegmetation.get_output_details()
width = input_details[0]['shape'][2]
height = input_details[0]['shape'][1]

#create pascal
colormap = create_pascal_label_colormap()

# Video Loop
cap = cv2.VideoCapture(1)
while(True):
    
    #get image
    #ret, frame = cap.read()
    frame         = cv2.imread("dog416.png")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    frame_resized_= np.array(frame_resized, dtype=np.float32)
    input_data = np.expand_dims(frame_resized_, axis=0)#input_data = input_data.swapaxes(1, 3)
    interpreterSegmetation.set_tensor(input_details[0]['index'], input_data) 

    #interpreter
    interpreterSegmetation.invoke()

    #prediction
    prediction  = interpreterSegmetation.get_tensor(output_details[0]['index'])
    mask = np.argmax(prediction, -1)[0].reshape((512, 512))

    # show result
    image_mask = colormap[mask]
    image_result =  cv2.addWeighted( frame_resized.astype('uint8'), 0.5, image_mask.astype('uint8'), 0.5, 0 )
    image_result = image_result + 64
    cv2.imshow('frame', image_result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



#prediction_swap = prediction.swapaxes(1, 3)#prediction_swap = prediction_swap.swapaxes(2,3)
'''
image_predict = prediction_swap[0][12]
image_predict = cv2.dilate(image_predict, np.ones((7,7), np.float32), iterations = 3)
image_zeros = np.zeros((512,512), dtype=np.float32)

while(True):
    image_rgb_merge = cv2.merge([image_zeros,image_zeros,image_predict*200])  
    image_result =  cv2.add( frame_resized.astype('float32'), image_rgb_merge.astype('float32') )
    cv2.imshow('frame', image_result.astype('uint8'))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
'''
