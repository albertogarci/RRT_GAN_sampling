#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from project.srv import roi, roiResponse

import os
import argparse

import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, ToPILImage, Normalize
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

##################################
####### ConvLeakyReLU block ######
##################################

class ConvLReLU(nn.Module):
    '''
    Conv2d + LeakyReLU

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size (default: int=4): kernel_size of Conv2d
        stride (default: int=2): stride of Conv2d
        padding (default: int=1): padding of Conv2d
        alpha (default: float=0.2): alpha of LeakyReLU
        
    '''
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size=4, stride=2, padding=1, alpha=0.2):
        super(ConvLReLU, self).__init__()
        self.conv_lrelu=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(alpha, inplace=True))
        
    def forward(self, x):
        return self.conv_lrelu(x)
    
##################################
######### ConvTanh block #########
##################################

class ConvTanh(nn.Module):
    '''
    Conv2d + Tanh

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size (default: int=1): kernel_size of Conv2d
        stride (default: int=1): stride of Conv2d
        padding (default: int=0): padding of Conv2d
        
    '''
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size=1, stride=1, padding=0):
        super(ConvTanh, self).__init__()
        self.conv_th=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.Tanh())
        
    def forward(self, x):
        return self.conv_th(x)
    
##################################
##### ConvBnLeakyReLU block ######
##################################

class ConvBnLReLU(nn.Module):
    '''
    Conv2d + BatchNorm2d + LeakyReLU

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size (default: int=4): kernel_size of Conv2d
        stride (default: int=2): stride of Conv2d
        padding (default: int=1): padding of Conv2d
        alpha (default: float=0.2): alpha of LeakyReLU
        
    '''
    def __init__(self, in_channels, out_channels,
                 kernel_size=4, stride=2, padding=1, alpha=0.2):
        super(ConvBnLReLU, self).__init__()
        self.conv_bn_lrelu=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            #nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(alpha, inplace=True))
        
    def forward(self, x):
        return self.conv_bn_lrelu(x)
    
##################################
####### UpConvBnReLU block #######
##################################

class UpConvBnReLU(nn.Module):
    '''
    ConvTranspose2d + InstanceNorm2d (replaced BatchNorm2d) + ReLU

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size (default: int=4): kernel_size of Conv2d
        stride (default: int=2): stride of Conv2d
        padding (default: int=1): padding of Conv2d
        
    '''
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size=4, stride=2, padding=1):
        super(UpConvBnReLU, self).__init__()
        self.upconv_bn_relu=nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            #nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU())
        
    def forward(self, x):
        return self.upconv_bn_relu(x)
class Generator(nn.Module):
    '''
    ROI Generator

    Args:
        map_channels (default: int=3): Number of Map input channels 
        point_channels (default: int=3): Number of Point input channels 
        hid_channels (default: int=32): Number of hidden channels
        out_channels (default: int=3): Number of output (ROI) channels
    '''
    def __init__(self,
                 map_channels=3, 
                 point_channels=3,
                 hid_channels=32,
                 out_channels=3):
        super(Generator, self).__init__()
        self.InputMap=ConvLReLU(map_channels, hid_channels//2, kernel_size=4)
        self.InputPoint=ConvLReLU(point_channels, hid_channels//2, kernel_size=4)
        
        self.DownBlock1=ConvBnLReLU(hid_channels, 2*hid_channels, kernel_size=4)
        self.DownBlock2=ConvBnLReLU(2*hid_channels, 4*hid_channels, kernel_size=4)
        self.DownBlock3=ConvBnLReLU(4*hid_channels, 8*hid_channels, kernel_size=4)
        self.DownBlock4=ConvBnLReLU(8*hid_channels, 8*hid_channels, kernel_size=4)
        
        self.UpBlock5=nn.Sequential(
                UpConvBnReLU(8*hid_channels, 8*hid_channels, kernel_size=4),
                nn.Dropout2d(0.5))
        self.UpBlock4=nn.Sequential(
                UpConvBnReLU(16*hid_channels, 4*hid_channels, kernel_size=4),
                nn.Dropout2d(0.5))
        self.UpBlock3=UpConvBnReLU(8*hid_channels, 2*hid_channels, kernel_size=4)
        self.UpBlock2=UpConvBnReLU(4*hid_channels, hid_channels, kernel_size=4)
        self.UpBlock1=UpConvBnReLU(2*hid_channels, 2*out_channels, kernel_size=4)
        
        self.Output=ConvTanh(4*out_channels, out_channels, kernel_size=1, stride=1)
        
    def forward(self, maps, points):
        x0 = torch.cat([maps, points], 1)
        m = self.InputMap(maps)
        p = self.InputPoint(points)
        x1 = torch.cat([m, p], 1)
        x2 = self.DownBlock1(x1)
        x3 = self.DownBlock2(x2)
        x4 = self.DownBlock3(x3)
        x5 = self.DownBlock4(x4)
        
        y5 = self.UpBlock5(x5)
        y4 = self.UpBlock4(torch.cat([y5, x4], 1))
        y3 = self.UpBlock3(torch.cat([y4, x3], 1))
        y2 = self.UpBlock2(torch.cat([y3, x2], 1))
        y1 = self.UpBlock1(torch.cat([y2, x1], 1))
        
        y0 = self.Output(torch.cat([y1, x0], 1))
        return y0

device = torch.device('cpu')

map = "map_94.png"
map = "cave.png"

map = Image.open("/home/osboxes/catkin_ws/src/project/" + map)
map = map.convert('RGB')

weights_file = "/home/osboxes/catkin_ws/src/project/checkpoint/model2.pt"
generator = Generator().to(device)
#generator.load_state_dict(torch.load(weights_file))

generator.load_state_dict(torch.load(weights_file, map_location=torch.device('cpu')))
points = [[1,2], [20,20]]

transform = Compose([ToTensor(),
                    Normalize(mean=(0.5, 0.5, 0.5),
                                std=(0.5, 0.5, 0.5))])


def rgb2binary(img):
    return (img[..., :] > 150).astype(float)


def process_image(load_dir):
    img = Image.open(load_dir).convert('RGB')
    data = rgb2binary(np.array(img))
    return data

def roi_from_image(load_dir):
    roi_data = process_image(load_dir)
    mask = roi_data[..., 0] * roi_data[..., 2]
    roi = list(zip(*np.where(mask == 0)))
    return roi

def predict(generator, map, points):
    #print(map.size )
    w, h = map.size 
    #map = transforms.ToTensor()(map).unsqueeze_(0)


    cstart = np.array([0., 0., 1.])
    cgoal = np.array([1., 0., 0.])

    task_map = np.ones((h, w, 3)) * 1.
    task_map[points[0][1], points[0][0]] = cstart
    task_map[points[1][1], points[1][0]] = cgoal

    map = transform(map)
    task_map = transform(task_map)

    map = map.unsqueeze_(0).to(device)
    task_map = task_map.unsqueeze_(0).to(device)
    # Map Discriminator`s loss
    fake_roi = generator(map.float(), task_map.float()).detach().cpu()
    out = np.squeeze(fake_roi + (map.cpu()-1) + (task_map.cpu()-1))
    img = np.transpose(out, (1,2,0))
    roi = np.squeeze(fake_roi)
    roi = np.array(np.transpose(roi, (1,2,0)) * 255).astype(np.uint8)

    #print(map.cpu().shape)
    roi_path = '/home/osboxes/catkin_ws/src/project/roi.png'

    import os
    if os.path.exists(roi_path):
        os.remove(roi_path)
        
    plt.imsave(roi_path, roi)
    return roi_from_image(roi_path)


def handle_roi(req):
    #print(req)
    points = np.array([[req.start_x,req.start_y], [req.goal_x,req.goal_y]])
    points = points - [-8.,8.]

    points = (np.absolute(points) * 4).astype(int)
    #print(points)

    roi = predict(generator, map, points)
    flattened = list(sum(roi, ()))

    #print(flattened)

    return roiResponse(flattened)

def roi_server():
    rospy.init_node("ROI_server")
    s = rospy.Service("ROI", roi, handle_roi)
    print("Ready to process ROI")
    rospy.spin()


if __name__ == "__main__":
    roi_server()