from PIL import ImageGrab
import cv2
import keyboard
import mouse
import numpy as np
#capture = cv2.VideoCapture(0)
wh_target = 320
confidence_threshold = 0.8
nms_threshold = 0.3

class_file = 'class.names'
class_names = []

with open(class_file, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

# tiny weights (higher frame rate, lower accuracies)
#model_config = 'yolov3-tiny.cfg'
#model_weights = 'yolov3-tiny.weights'

# using more trained weights (lower frame rate, higher accuracies)
model_config = 'yolov3.cfg'
model_weights = 'yolov3.weights'

network = cv2.dnn.readNetFromDarknet(model_config, model_weights)

# use CPU
network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# use GPU (CUDA)
#network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

def find_objects(outputs, img):
    hT, wT, cT = img.shape
    bounding_box = []
    class_ids = []
    confidence_values = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                width, height = int(detection[2] * wT), int(detection[3] * hT)
                x, y = int(detection[0] * wT - width / 2), int(detection[1] * hT - height / 2)
                bounding_box.append([x, y, width, height])
                class_ids.append(class_id)
                confidence_values.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bounding_box, confidence_values, confidence_threshold, nms_threshold)

    for i in indices:

        i = i[0]
        box = bounding_box[i]
        print(class_names[class_ids[i]])
        #print(type(box))
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (100, 100, 255), 2)
        cv2.putText(img, f'{class_names[class_ids[i]]} {int(confidence_values[i] * 100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)

def set_roi():
    global ROI_SET, x1, y1, x2, y2
    ROI_SET = False
    print("Select your ROI using mouse drag.")
    while(mouse.is_pressed() == False):
        x1, y1 = mouse.get_position()
        while(mouse.is_pressed() == True):
            x2, y2 = mouse.get_position()
            while(mouse.is_pressed() == False):
                print("Your ROI : {0}, {1}, {2}, {3}".format(x1, y1, x2, y2))
                ROI_SET = True
                return

keyboard.add_hotkey("ctrl+1", lambda: set_roi())

ROI_SET = False
x1, y1, x2, y2 = 0, 0, 0, 0
while True:
    if ROI_SET == True:
        image = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(x1, y1, x2, y2))), cv2.COLOR_BGR2RGB)
        #cv2.imshow("image", image)
        blob = cv2.dnn.blobFromImage(image, 1 / 255, (wh_target, wh_target), [0, 0, 0], 1, crop=False)
        network.setInput(blob)

        layer_names = network.getLayerNames()
        output_names = [layer_names[i[0] - 1] for i in network.getUnconnectedOutLayers()]

        outputs = network.forward(output_names)

        find_objects(outputs, image)

        cv2.imshow('Camera', image)
        key = cv2.waitKey(100)
        if key == ord("q"):
            print("Quit")
            break

cv2.destroyAllWindows()