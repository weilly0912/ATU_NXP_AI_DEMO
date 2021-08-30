import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter 
#https://github.com/tantara/JejuNet

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


class_names = get_classes("voc_classes.txt")

# --------------------------------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------------------------------
interpreterSegmetation = Interpreter(model_path='deepv3_mnv2_pscal_128_enrp_byWeilly.tflite')

interpreterSegmetation.allocate_tensors() 
input_details = interpreterSegmetation.get_input_details()
output_details = interpreterSegmetation.get_output_details()
width = input_details[0]['shape'][2]
height = input_details[0]['shape'][1]


#cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture("v4l2src device=/dev/video1 ! video/x-raw,format=YUY2,width=1280,height=720,framerate=30/1! videoscale!videoconvert ! appsink")
while(True):

    ret, frame = cap.read() #480x640,frame = cv2.resize(frame, (640, 480))=> no better
    
    #frame         = cv2.imread("dog416.png")
    #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame, (width, height))#
    frame_resized_= np.array(frame_resized, dtype=np.uint8)
    input_data = np.expand_dims(frame_resized_, axis=0)
    interpreterSegmetation.set_tensor(input_details[0]['index'], input_data) 

    #interpreter
    interpreterSegmetation.invoke()

    #prediction
    prediction  = interpreterSegmetation.get_tensor(output_details[0]['index']) #fPixel = class
    prediction  = np.reshape(prediction, [width*height])
    prediction_img  = np.reshape(prediction, [width, height])

    #index
    labelIndex = np.argmax(prediction)
    labelbinCnt= np.bincount(prediction)
    labelcount = np.unique(np.sort(labelbinCnt)) 
    labelcount_len = len(labelcount)

    #show
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

    image_rgb_merge =  cv2.resize(image_rgb_merge, (frame.shape[1], frame.shape[0]))
    image_result =  cv2.add( frame.astype('uint8'), image_rgb_merge.astype('uint8') )
    print("Class of Num = ", labelcount_len , ' , ' ,"class of MAX = ", class_names[np.where(labelbinCnt==labelcount[labelcount_len-2])[0][0] - 1])

    cv2.imshow('frame', image_result.astype('uint8'))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

