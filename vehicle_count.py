import cv2
import csv
import collections
import numpy as np
import time
from tracker import *
tracker = EuclideanDistTracker()
cap = cv2.VideoCapture('test3.mp4')
input_size = 320
confThreshold = 0.2
nmsThreshold = 0.2
font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 2
middle_line_position = 225
up_line_position = middle_line_position - 15
down_line_position = middle_line_position + 15

classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')
print(classNames)
print(len(classNames))

required_class_index = [2, 3, 5, 7]
detected_classNames = []

modelConfiguration = 'yolo-320.cfg'
modelWeigheights = 'yolo-320.weights'
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')

def find_center(x, y, w, h):
    cx = x + int(w/2)
    cy = y + int(h/2)
    return cx, cy


temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0]
down_list = [0, 0, 0, 0]


vehicle_speeds = {}  

def estimate_speed(prev_center, current_center, time_diff, px_to_meters_ratio):
     if time_diff > 0:
      
        distance_px = np.linalg.norm(np.array(prev_center) - np.array(current_center))
        distance_m = distance_px * px_to_meters_ratio 
        speed_m_per_s = distance_m / time_diff 
        speed_kmh = speed_m_per_s * 3.6 
        return speed_kmh
     else:
        return 0 
    


def count_vehicle(box_id, img, frame_time):
    x, y, w, h, id, index = box_id
    center = find_center(x, y, w, h)
    ix, iy = center

    
    if id in vehicle_speeds:
        prev_center, prev_time = vehicle_speeds[id]
        time_diff = frame_time - prev_time
        px_to_meters_ratio = 0.05 
        speed = estimate_speed(prev_center, center, time_diff, px_to_meters_ratio)
        
     
        cv2.putText(img, f'Speed: {int(speed)*5} km/h', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        
        vehicle_speeds[id] = (center, frame_time)
    else:
        
        vehicle_speeds[id] = (center, frame_time)

   
    if (iy > up_line_position) and (iy < middle_line_position):
        if id not in temp_up_list:
            temp_up_list.append(id)
    elif iy < down_line_position and iy > middle_line_position:
        if id not in temp_down_list:
            temp_down_list.append(id)
    elif iy < up_line_position:
        if id in temp_down_list:
            temp_down_list.remove(id)
            up_list[index] += 1
    elif iy > down_line_position:
        if id in temp_up_list:
            temp_up_list.remove(id)
            down_list[index] += 1

    
    cv2.circle(img, center, 2, (0, 0, 255), -1)


def postProcess(outputs, img, frame_time):
    global detected_classNames
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []

  
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index and confidence > confThreshold:
                w, h = int(det[2]*width), int(det[3]*height)
                x, y = int((det[0]*width) - w/2), int((det[1]*height) - h/2)
                boxes.append([x, y, w, h])
                classIds.append(classId)
                confidence_scores.append(float(confidence))

   
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        color = [int(c) for c in colors[classIds[i]]]
        name = classNames[classIds[i]]
        detected_classNames.append(name)

      
        cv2.putText(img, f'{name.upper()} {int(confidence_scores[i]*100)}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        detection.append([x, y, w, h, required_class_index.index(classIds[i])])

   
    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        count_vehicle(box_id, img, frame_time)

def realTime():
    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
        ih, iw, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

        
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)

        frame_time = time.time()

       
        postProcess(outputs, img, frame_time)

        
        cv2.line(img, (0, middle_line_position), (iw, middle_line_position), (255, 0, 255), 2)
        cv2.line(img, (0, up_line_position), (iw, up_line_position), (0, 0, 255), 2)
        cv2.line(img, (0, down_line_position), (iw, down_line_position), (0, 0, 255), 2)

        cv2.putText(img, "Up", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Down", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Car:        "+str(up_list[0])+"     "+ str(down_list[0]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Motorbike:  "+str(up_list[1])+"     "+ str(down_list[1]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Bus:        "+str(up_list[2])+"     "+ str(down_list[2]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Truck:      " + str(up_list[3]) + "     " + str(down_list[3]), (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

        
        cv2.imshow('Vehicle Detection', img)

       
        if cv2.waitKey(1) & 0xFF == 27:
            break

   
    cap.release()
    cv2.destroyAllWindows()


realTime()