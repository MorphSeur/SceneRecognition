# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
import cv2
import numpy as np
import time


# th architecture to use
arch = 'resnet50'
# load the pre-trained weights
model_file = './models/%s_places365.pth.tar' % arch
model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()

# load the class label
file_name = './categories/categories_places365.txt'
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)


# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# cap = cv2.VideoCapture('.\\imgs\\sample.mp4')  
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
frames = 0
lblDic = {}
start = time.time() 
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    cv2_im = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2_im)
   
    input_img = V(centre_crop(img).unsqueeze(0))
    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    preds = ''
    horPos = 10
    verPos = 50
    label = [] 
    # output the prediction
    for i in range(0, 5):
        preds = "{} = {:.3f}".format(classes[idx[i]],probs[i])
        label.append({classes[idx[i]]:"{:.3f}".format(probs[i])})
        cv2.putText(frame,preds,(horPos,verPos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
        verPos += 30
    print(label)
    lblDic[frames] = label
    frames += 1
    print ("Frames : "+ str(frames))
    print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

import json
with open('places.json', 'w') as outfile:
    json.dump(lblDic, outfile)


# load the test image
#img_name = './imgs/12.jpg'
#img = Image.open(img_name)
#input_img = V(centre_crop(img).unsqueeze(0))
# forward pass
#logit = model.forward(input_img)
#h_x = F.softmax(logit, 1).data.squeeze()
#probs, idx = h_x.sort(0, True)

#print('{} prediction on {}'.format(arch,img_name))
# output the prediction
#for i in range(0, 5):
#    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
