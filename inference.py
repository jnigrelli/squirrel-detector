import cv2
import PIL

import torch
from torchvision.io import read_image

import transforms as T

import time

time_start = time.time()
image = read_image("squirrel_eval.jpg")
image = T.ConvertImageDtype(torch.float32)(image)[0]
print("Load and convert image time: " + str((time.time() - time_start)))

time_start = time.time()
path = "squirrel_detector_v3.pt"

# mps device doesn't work for some operations...
device = 'cpu'

model_test = torch.load(path, map_location=torch.device(device))
model_test.eval()
print("Load and convert model time: " + str((time.time() - time_start)))

model_test = torch.compile(model_test)

image = image.to(device)

time_start = time.time()
pred = model_test([image])
print("Inference time: " + str((time.time() - time_start)))
print(pred[0]["boxes"])
print(pred[0]["scores"])