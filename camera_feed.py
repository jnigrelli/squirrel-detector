import cv2
import PIL

import torch
from torchvision.io import read_image

import transforms as T
import torchvision.transforms.functional as transform

from serial import Serial
import time

import numpy as np

def predict_frame(frame, model):
    frame = [frame.clone().detach()]
    pred = model(frame)[0]
    boxes = pred["boxes"]
    scores = pred["scores"]
    return boxes, scores

# frame needs to be a CV2 IMAGE
def plot_boxes(boxes, scores, frame, threshold=0.9):
    n = len(scores)

    for i in range(n):
        row = boxes[i]
        # If score is less than 0.2 we avoid making a prediction.
        if scores[i] < threshold: 
            continue

        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])

        bgr = (0, 255, 0) # color of the box
         # Get the name of label index
        label_font = cv2.FONT_HERSHEY_SIMPLEX #Font for the label.
        cv2.rectangle(frame, \
                      (x1, y1), (x2, y2), \
                       bgr, 2) #Plot the boxes
        cv2.putText(frame,\
                    "squirrel: " + str(scores[i].item()), \
                    (x1, y1), \
                    label_font, 0.9, bgr, 2) #Put a label over box.
    return frame

#SERIAL PORT/MONITOR SETUP
use_serial = True
try:
    arduino = Serial(port='/dev/cu.usbmodem101', baudrate=115200, timeout=0)
except:
    use_serial=False

def send_squirrel_detected():
    arduino.write(bytes('1', 'utf-8'))
    time.sleep(0.05)
    data = arduino.readline()
    return data

path = "squirrel_detector_v9.pt"

device = 'cpu'

model = torch.load(path, map_location=torch.device(device))
model.eval()

vs = cv2.VideoCapture(0)

while True:
    (grabbed, frame) = vs.read()
 
    # check there's any frame to be grabbed from the steam
    if not grabbed:
        break
 
    # clone the current frame, convert it from BGR into RGB
    output = frame.copy()
    tensor_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    tensor_image = transform.to_tensor(frame) 
    boxes, scores = predict_frame(tensor_image, model)
    
    threshold = 0.8
    if use_serial:
        for score in scores:
            if score > threshold:
                msg = send_squirrel_detected()
                print(msg)
                break

    plotted_frame = plot_boxes(boxes, scores, frame, threshold=threshold)
    cv2.imshow('image', plotted_frame)
    cv2.waitKey(1)