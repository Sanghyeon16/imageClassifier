import json
import pandas as pd
import numpy as np
import torch
from torchvision import transforms, datasets, models
from torch import nn, optim
from torch.nn import functional as F

# from PIL import Image
# import matplotlib.pyplot as plt 

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
    parser.add_argument('data_directory', help='image data directory')
    parser.add_argument('--save_dir', help='save model directory')
    parser.add_argument('--arch', help='base model vgg13')
    parser.add_argument('--learning_rate', help='learning rate')
    parser.add_argument('--hidden_units',default=4096, help='dimension of hidden layer')
    parser.add_argument('--epochs', help='epochs')
    parser.add_argument('--gpu', help='gpu')
    args = parser.parse_args()
    return args

def data_preprocessing(data_dir):
    logger.info("start data preprocess")
    stime = time.time()
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    test_data_transforms = transforms.Compose([transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])



    image_datasets = {
                    'train_image':datasets.ImageFolder(train_dir, transform=train_data_transforms),
                    'valid_image':datasets.ImageFolder(valid_dir, transform=test_data_transforms),
                    'test_image':datasets.ImageFolder(test_dir, transform=test_data_transforms)
                    }

    dataloaders = {
        'train_dataloader':torch.utils.data.DataLoader(image_datasets['train_image'], batch_size=64, shuffle=True),
        'valid_dataloader':torch.utils.data.DataLoader(image_datasets['valid_image'], batch_size=64),
        'test_dataloader':torch.utils.data.DataLoader(image_datasets['test_image'], batch_size=64)
                    }
    etime = time.strftime('%H:%M:%S',time.gmtime(time.time()-stime))
    logger.info(f"data preprocessing time: {etime}")
    return dataloaders, image_datasets



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
   
def build_model(arch, hidden_units):
    model = models.vgg13(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    input_dim = 25088
    output_dim = 102
    num_layers = 3
    dropout_prob = 0.2
    logger.info(f"hidden_dim:{int(hidden_units)}")
    model.classifier = imageClassifier(input_dim, output_dim, int(hidden_units), num_layers, dropout_prob)
    return model

def train(logger, model, dataloaders, device):
    logger.info("initiate training process")
    
    logger.info(f"device:{device}")
    learning_rate = float(args.learning_rate)
    epochs = int(args.epochs)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    model.to(device)
    # training starts
    logger.info("start training process")
    stime = time.time()
    steps = 0
    running_loss = 0
    print_every = 50
    for epoch in range(epochs):
        for inputs, labels in dataloaders['train_dataloader']:
            steps += 1
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                valid_acc = 0
                valid_size = len(dataloaders['train_dataloader'])
                with torch.no_grad():
                    for inputs, labels in dataloaders['train_dataloader']:
                        
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outputs = model.forward(inputs)
                        loss = criterion(outputs, labels)
                        valid_loss += loss.item()
                        ps = torch.exp(outputs).data
                        comparison = labels.data == ps.max(1)[1]
                        valid_acc += torch.mean(comparison.type(torch.FloatTensor)).item()
                logger.info(f"Epoch {epoch+1}/{epochs} | Training Loss: {running_loss/print_every}, Valid Loss: {valid_loss/valid_size}, Valid Accuracy: {valid_acc/valid_size}")
                running_loss = 0
                model.train()
    etime = time.strftime('%H:%M:%S',time.gmtime(time.time()-stime))
    logger.info(f"training time: {etime}")
    return model

def test_model(logger, model, dataloaders, device):
    logger.info("start testing process")
    stime = time.time()
    model.to(device)
    test_acc = 0
    test_size = len(dataloaders['test_dataloader'])
    with torch.no_grad():
        for inputs, labels in dataloaders['test_dataloader']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            ps = torch.exp(outputs).data
            comparison = labels.data == ps.max(1)[1]
            test_acc += torch.mean(comparison.type(torch.FloatTensor))
            
        print(f"Test Accuracy: {100*test_acc/test_size}%")
    etime = time.strftime('%H:%M:%S',time.gmtime(time.time()-stime))
    logger.info(f"training time: {etime}")
    
def save_model(logger,model, image_datasets, hidden_units, save_directory):
    model.class_to_idx = image_datasets['train_image'].class_to_idx
    model.cpu
    torch.save({'structure' :'vgg13',
            'hidden_layer1': hidden_units,
            'state_dict':model.state_dict(),
            'class_to_idx':model.class_to_idx},
            save_directory)
    
if __name__ == '__main__':
    args = parse_args('imageClassifier')
    stime = time.time()
    logger = init_logger('imageClassifier')
    dataloaders, image_datasets = data_preprocessing(args.data_directory)
    print('data_preprocessing')
    model = build_model(args.arch, args.hidden_units)
    print('build_model')
    if args.gpu == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    model = train(logger, model, dataloaders, device)
    print('train_model')
    test_model(logger, model, dataloaders, device)
    print('test_model')
    save_model(logger,model,image_datasets, args.hidden_units, args.save_dir)
    print('save_model')
    