'''
USAGE:
python test.py --model ../outputs/sports.pth --label-bin ../outputs/lb.pkl --input ../input/example_clips/chess.mp4 --output ../outputs/chess.mp4
'''
import argparse
import time

import albumentations
import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

import cnn_models

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True,
    help="path to trained serialized model")
ap.add_argument('-l', '--label-bin', required=True,
    help="path to  label binarizer")
ap.add_argument('-i', '--input', required=True,
    help='path to our input video')
ap.add_argument('-o', '--output', required=True, type=str,
    help='path to our output video')
args = vars(ap.parse_args())

# load the trained model and label binarizer from disk
print('Loading model and label binarizer...')
lb = joblib.load(args['label_bin'])
try:
  model = cnn_models.CustomCNN().cuda()
except:
  model = cnn_models.CustomCNN().cpu()
print('Model Loaded...')
model.load_state_dict(torch.load(args['model']))
print('Loaded model state_dict...')
aug = albumentations.Compose([
    albumentations.Resize(224, 224),
    ])


cap = cv2.VideoCapture(args['input'])

if (cap.isOpened() == False):
    print('Error while trying to read video. Plese check again...')

# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# define codec and create VideoWriter object
out = cv2.VideoWriter(str(args['output']), cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width,frame_height))

# read until end of video
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        model.eval()
        with torch.no_grad():
            # conver to PIL RGB format before predictions
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_image = aug(image=np.array(pil_image))['image']
            pil_image = np.transpose(pil_image, (2, 0, 1)).astype(np.float32)
            #todo need to update cuda
            pil_image = torch.tensor(pil_image, dtype=torch.float).cpu()
            pil_image = pil_image.unsqueeze(0)
            
            outputs = model(pil_image)
            _, preds = torch.max(outputs.data, 1)
        
        cv2.putText(frame, lb.classes_[preds], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
        cv2.imshow('image', frame)
        out.write(frame)
        # press `q` to exit
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
    else: 
        break
# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()
