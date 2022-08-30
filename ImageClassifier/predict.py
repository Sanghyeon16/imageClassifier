import json
import pandas as pd
import numpy as np
import torch
from torchvision import transforms, datasets, models
from torch import nn, optim
from torch.nn import functional as F

from PIL import Image
import matplotlib.pyplot as plt 

import os
import time
import argparse
import logging
import sys

def init_logger(project_name):
    logger = logging.getLogger(project_name)
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stream_handler)
    return logger
   
def parse_args(project_name):
    parser = argparse.ArgumentParser(description=project_name)
    parser.add_argument('input_path', help='image data directory')
    parser.add_argument('checkpoint', help='saved model directory')
    parser.add_argument('--category_names', help='encoder')
    parser.add_argument('--topk', help='number of predictions')
    parser.add_argument('--gpu', help='gpu')
    args = parser.parse_args()
    return args




class imageClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers=1, drop_prob=0.3):
        super().__init__()
        
        self.n_layers = n_layers
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.dropout_layer = nn.Dropout(p=drop_prob)
        self.output_layer = nn.Linear(int(hidden_dim/2), output_dim)
        
    def forward(self,x):
        h = self.input_layer(x)
        h = F.relu(h)
        h = self.dropout_layer(h)
        h = self.hidden_layer(h)
        h = F.relu(h)
        h = self.dropout_layer(h)
        y = F.log_softmax(self.output_layer(h),dim=1)
        return y

def load_model(path):
    checkpoint = torch.load(path)
    model = models.vgg13(pretrained=True)
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']
    
    input_dim = 25088
    output_dim = 102
    num_layers = 3
    dropout_prob = 0.2
    hidden_units = int(hidden_layer1)
    
    model.classifier = imageClassifier(input_dim, output_dim, hidden_units, num_layers, dropout_prob)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(path):
    img = Image.open(path)
    w, h = img.size
    size = [w,h]
    max_frame = max(size)
    larger_axis = size.index(max_frame)
    smaller_axis = 0 if larger_axis else 1
    ratio = size[larger_axis]/size[smaller_axis]

    rescale = [0,0]
    rescale[larger_axis] = int(256*ratio)
    rescale[smaller_axis] = 256
    img = img.resize(rescale)
    w, h = rescale
    l = (w-244)/2
    r = (w+244)/2
    t = (h-244)/2
    b = (h+244)/2
    img = img.crop((l,t,r,b))
    np_image = np.array(img).astype('float64') / [255,255,255]
    np_image = (np_image - [0.485,0.456,0.406])/ [0.229,0.224,0.225]
    np_image = np_image.transpose((2,0,1))
    return np_image
    
def predict(image_path, model, topk, device):
    model.to(device)
    with torch.no_grad():
        img = process_image(image_path)
        img = torch.from_numpy(img)
        img.unsqueeze_(0)
        img = img.float()
        model.eval()
        outputs = model.forward(img.to(device))
        probs, preds = torch.exp(outputs).topk(topk)
        return probs[0].tolist(), preds[0].add(1).tolist()

def print_results(img_path, model, decoder_path, topk, device):
    
    with open(decoder_path, 'r') as f:
        cat_to_name = json.load(f)
    probs, preds = predict(img_path, model, topk, device)
    label2name = [cat_to_name[str(pred)] + f"({str(pred)})" for pred in preds]
    print(f"Top {topk} predictions: {label2name}")
    return None
    
if __name__ == '__main__':
    args = parse_args('imageClassifier')
    stime = time.time()
    logger = init_logger('imageClassifier')
    model = load_model(args.checkpoint)
    print('load_model')
    if args.gpu == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    probs, results = predict(args.input_path, model, int(args.topk), device)
    print('prediction')
    print_results(args.input_path, model, args.category_names, int(args.topk), device)
    print('print_results')
    