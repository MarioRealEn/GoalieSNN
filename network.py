
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import torchvision.transforms as transforms
import snntorch as snn
from snntorch import surrogate
import torchvision.transforms.functional as TF
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, rayleigh
import time
import data as dt
from simulator import BALL_RADIUS, Camera, Field
from tqdm import tqdm
from statsmodels.stats.stattools import durbin_watson


MAX_VAL_R_CAM = dt.MAX_VAL_R_CAM
MAX_VAL_Y_CAM = dt.MAX_VAL_Y_CAM
MAX_VAL_Y_CAM = 720
BALL_RADIUS = BALL_RADIUS
CHUNK_SIZE = 40

W = 14.0      # field width (m)
L = 11.0       # field length (m)

field = Field(W, L, 2.4, 1)


# CameraLeft
cam_x = -0.05994 + 7
cam_y = -11 + 0.12 + 0.025 + 11
cam_z = 0.55

camera_pos = np.array([cam_x, cam_y, cam_z])          # Camera position in world space

fps = 200

# ---------------------
# Define camera parameters for the pinhole model
# ---------------------      
# Where the camera is looking (center of field)
camera_target = np.array([field.center[0], field.center[1], camera_pos[2]])        # Where the camera is looking (center of field) It would be good to transform this into an angle and assume verticality
orientation = Camera.get_orientation(camera_pos, camera_target)
img_width, img_height = 1280, 720               # Image resolution (in pixels)
focal_length = 0.008                             # Focal length in meters

CAMERA = Camera(camera_pos, orientation, focal_length, img_width, img_height, fps = fps)


class SCNN2LImageClassification(nn.Module):
    def __init__(self, image_shape, bins_factor, beta=0.8, learn_threshold = False):
        """
        input_shape: tuple (channels, height, width) of input event data.
        """
        super(SCNN2LImageClassification, self).__init__()
        self.name = 'Tracker2L'
        # Convolutional layers (assuming 2-channel input for polarity split)
        self.beta = beta
        self.learn_threshold = learn_threshold
        self.image_shape = image_shape
        self.max_values = {
            'X': self.image_shape[2],
            'Y': self.image_shape[1],
            'Radius': MAX_VAL_R_CAM,
            'x_cam': self.image_shape[2],
            'y_cam': MAX_VAL_Y_CAM,
            'R_cam': MAX_VAL_R_CAM
        }

        # Convolutional layers (assuming 2-channel input for polarity split)
        channels, y_pixels, x_pixels = image_shape
        x_bins, y_bins = int(x_pixels*bins_factor), int(y_pixels*bins_factor)
        input_shape = (channels, y_pixels, x_pixels)
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=1, padding=2)   # Expected: from (2, H, W) to (16, H/2, W/2)
        self.lif1 = snn.Leaky(beta)
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Expected: (32, H/4, W/4)
        self.lif2 = snn.Leaky(beta)
        self.mp2 = nn.MaxPool2d(2)

        # Dynamically compute the flattened feature size by passing a dummy input.
        self.flattened_size = self._get_flattened_size(input_shape)
        print(f"Flattened feature size: {self.flattened_size}")
        
        # Two fully connected branches: one for x-coordinate, one for y-coordinate
        self.fc_x = nn.Linear(self.flattened_size, x_bins)
        self.lif_x = snn.Leaky(beta)
        self.fc_y = nn.Linear(self.flattened_size, y_bins)
        self.lif_y = snn.Leaky(beta)
        
        # Surrogate gradient activation (spiking behavior simulation) IDK WHAT THIS IS
        # self.spike_fn = surrogate.fast_sigmoid(slope=25)

    def _get_flattened_size(self, input_shape):
        # Create a dummy input tensor with the given shape.
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            x = self.conv1(x)
            x = self.mp1(x)
            x = self.conv2(x)
            x = self.mp2(x)
            flattened_size = x.view(1, -1).size(1)
        return flattened_size

    def forward(self, x, num_steps=10):
        """
        x: input tensor of shape [batch, channels, height, width]
        num_steps: number of simulation time steps (simulate repeated evaluation to mimic spiking dynamics)
        """
        # Initialize hidden states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_x = self.lif_x.init_leaky()
        mem_y = self.lif_y.init_leaky()

        # Record final layer
        spk_x_rec = []
        spk_y_rec = []
        mem_x_rec = []
        mem_y_rec = []

        batch_size = x.size(0)
        outputs_x = 0
        outputs_y = 0
        for step in range(num_steps):
            # Convolutional layers with spiking activations
            x1 = self.conv1(x)
            spk1, mem1 = self.lif1(self.mp1(x1), mem1)
            
            x2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(self.mp2(x2), mem2)
            
            # Flatten features
            s2_flat = spk2.view(batch_size, -1)
            # Fully connected branches for x and y
            out_x = self.fc_x(s2_flat)
            spk_x = self.lif_x(out_x, mem_x)
            out_y = self.fc_y(s2_flat)
            spk_y = self.lif_y(out_y, mem_y)
            # Record spiking activity and membrane potential for the final layer
            spk_x_rec.append(spk_x)
            spk_y_rec.append(spk_y)
            mem_x_rec.append(mem_x)
            mem_y_rec.append(mem_y)
            outputs_x += out_x
            outputs_y += out_y
        # Average over time steps
        outputs_x = outputs_x / num_steps
        outputs_y = outputs_y / num_steps
        
        # Apply softmax to get probabilities
        # probs_x = F.softmax(outputs_x, dim=1)
        # probs_y = F.softmax(outputs_y, dim=1)
        
        return outputs_x, outputs_y#, [spk_x_rec, spk_y_rec, mem_x_rec, mem_y_rec]
    
    def start_training(self, trainloader, optimizer, device, validationloader = None, num_steps = 10, num_epochs=20, plot = True):
        batch_size = trainloader.batch_size
        loss_function = classification_loss
        self.training_params = {
            "type": self.name,
            "batch_size": batch_size,
            "num_steps": num_steps,
            "loss_function": loss_function.__name__,
            "optimizer": optimizer,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "num_epochs": num_epochs,
            "quantization": trainloader.dataset.quantization,
            "label_quantization": trainloader.dataset.label_quantization,
            "beta": self.beta,
            "learn_threshold": self.learn_threshold,
            "image_shape": self.image_shape,
            }
        training_loop_images(self, trainloader, optimizer, device, loss_function, validationloader, num_steps, num_epochs, plot=plot)
    
    def evaluate(self, testloader, device, num_steps, print_results=False):
        return evaluate_classification_tracker(self, testloader, device, num_steps, print_results)

class SCNN2LImageRegression(nn.Module):
    def __init__(self, image_shape, beta=0.8, learn_threshold = False):
        """
        input_shape: tuple (channels, height, width) of input event data.
        """
        super(SCNN2LImageRegression, self).__init__()
        self.name = 'Tracker2L'
        # Convolutional layers (assuming 2-channel input for polarity split)
        self.beta = beta
        self.learn_threshold = learn_threshold
        self.image_shape = image_shape
        self.max_values = {
            'X': self.image_shape[2],
            'Y': self.image_shape[1],
            'Radius': MAX_VAL_R_CAM,
            'x_cam': self.image_shape[2],
            'y_cam': MAX_VAL_Y_CAM,
            'R_cam': MAX_VAL_R_CAM
        }

        # Convolutional layers (assuming 2-channel input for polarity split)
        channels, y_pixels, x_pixels = image_shape
        self.input_shape = (channels, y_pixels, x_pixels)
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=1, padding=2)   # Expected: from (2, H, W) to (16, H/2, W/2)
        self.lif1 = snn.Leaky(beta)
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Expected: (32, H/4, W/4)
        self.lif2 = snn.Leaky(beta)
        self.mp2 = nn.MaxPool2d(2)

        # Dynamically compute the flattened feature size by passing a dummy input.
        self.flattened_size = self._get_flattened_size(self.input_shape)
        print(f"Flattened feature size: {self.flattened_size}")
        
        # Two fully connected branches: one for x-coordinate, one for y-coordinate
        self.fc_x = nn.Linear(self.flattened_size, 1)
        self.lif_x = snn.Leaky(beta)
        self.fc_y = nn.Linear(self.flattened_size, 1)
        self.lif_y = snn.Leaky(beta)
        
        # Surrogate gradient activation (spiking behavior simulation) IDK WHAT THIS IS
        # self.spike_fn = surrogate.fast_sigmoid(slope=25)

    def _get_flattened_size(self, input_shape):
        # Create a dummy input tensor with the given shape.
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            x = self.conv1(x)
            x = self.mp1(x)
            x = self.conv2(x)
            x = self.mp2(x)
            flattened_size = x.view(1, -1).size(1)
        return flattened_size

    def forward(self, x, num_steps=10):
        """
        x: input tensor of shape [batch, channels, height, width]
        num_steps: number of simulation time steps (simulate repeated evaluation to mimic spiking dynamics)
        """
        # Initialize hidden states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_x = self.lif_x.init_leaky()
        mem_y = self.lif_y.init_leaky()

        # Record final layer
        spk_x_rec = []
        spk_y_rec = []
        mem_x_rec = []
        mem_y_rec = []

        batch_size = x.size(0)
        outputs_x = 0
        outputs_y = 0
        for step in range(num_steps):
            # Convolutional layers with spiking activations
            x1 = self.conv1(x)
            spk1, mem1 = self.lif1(self.mp1(x1), mem1)
            
            x2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(self.mp2(x2), mem2)
            
            # Flatten features
            s2_flat = spk2.view(batch_size, -1)
            # Fully connected branches for x and y
            out_x = self.fc_x(s2_flat)
            spk_x = self.lif_x(out_x, mem_x)
            out_y = self.fc_y(s2_flat)
            spk_y = self.lif_y(out_y, mem_y)
            # Record spiking activity and membrane potential for the final layer
            spk_x_rec.append(spk_x)
            spk_y_rec.append(spk_y)
            mem_x_rec.append(mem_x)
            mem_y_rec.append(mem_y)
            outputs_x += out_x
            outputs_y += out_y
        # Average over time steps
        outputs_x = outputs_x / num_steps
        outputs_y = outputs_y / num_steps
        
        # Apply softmax to get probabilities
        # probs_x = F.softmax(outputs_x, dim=1)
        # probs_y = F.softmax(outputs_y, dim=1)
        
        return outputs_x, outputs_y#, [spk_x_rec, spk_y_rec, mem_x_rec, mem_y_rec]
    
    def start_training(self, trainloader, optimizer, device, validationloader = None, num_steps = 10, num_epochs=20, plot = True):
        batch_size = trainloader.batch_size
        loss_function = regression_loss
        self.training_params = {
            "type": self.name,
            "batch_size": batch_size,
            "num_steps": num_steps,
            "loss_function": loss_function.__name__,
            "optimizer": optimizer,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "num_epochs": num_epochs,
            "quantization": trainloader.dataset.quantization,
            "label_quantization": trainloader.dataset.label_quantization,
            "beta": self.beta,
            "learn_threshold": self.learn_threshold,
            "image_shape": self.image_shape,
            }
        training_loop_images(self, trainloader, optimizer, device, loss_function, validationloader, num_steps, num_epochs, plot=plot)
    
    def evaluate(self, testloader, device, num_steps, print_results=False):
        return evaluate_regression_tracker(self, testloader, device, num_steps, print_results)


class SCNN3LImageClassification(nn.Module):
    def __init__(self, image_shape, bins_factor, beta=0.8, learn_threshold = False):
        """
        input_shape: tuple (channels, height, width) of input event data.
        """
        super(SCNN3LImageClassification, self).__init__()

        self.name = 'ImageClassification'
        self.task = 'classification'
        self.beta = beta
        self.learn_threshold = learn_threshold
        self.image_shape = image_shape
        self.max_values = {
            'X': self.image_shape[2],
            'Y': self.image_shape[1],
            'Radius': MAX_VAL_R_CAM,
            'x_cam': self.image_shape[2],
            'y_cam': MAX_VAL_Y_CAM,
            'R_cam': MAX_VAL_R_CAM
        }

        # Convolutional layers (assuming 2-channel input for polarity split)
        channels, y_pixels, x_pixels = image_shape
        x_bins, y_bins = int(x_pixels*bins_factor), int(y_pixels*bins_factor)

        self.input_shape = (channels, y_pixels, x_pixels)
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=1, padding=2)   # Expected: from (2, H, W) to (16, H/2, W/2)
        self.lif1 = snn.Leaky(beta, learn_threshold=learn_threshold)
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Expected: (32, H/4, W/4)
        self.lif2 = snn.Leaky(beta, learn_threshold=learn_threshold)
        self.mp2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Expected: (64, H/8, W/8)
        self.lif3 = snn.Leaky(beta, learn_threshold=learn_threshold)
        self.mp3 = nn.MaxPool2d(2)

        # Dynamically compute the flattened feature size by passing a dummy input.
        self.flattened_size = self._get_flattened_size(self.input_shape)
        print(f"Flattened feature size: {self.flattened_size}")
        
        # Two fully connected branches: one for x-coordinate, one for y-coordinate
        self.fc_x = nn.Linear(self.flattened_size, x_bins)
        self.fc_y = nn.Linear(self.flattened_size, y_bins)
        print(f"Number of x and y bins: {x_bins}, {y_bins}")
        
        # Surrogate gradient activation (spiking behavior simulation) IDK WHAT THIS IS
        # self.spike_fn = surrogate.fast_sigmoid(slope=25)

    def _get_flattened_size(self, input_shape):
        # Create a dummy input tensor with the given shape.
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            x = self.conv1(x)
            x = self.mp1(x)
            x = self.conv2(x)
            x = self.mp2(x)
            x = self.conv3(x)
            x = self.mp3(x)
            flattened_size = x.view(1, -1).size(1)
        return flattened_size

    def forward(self, x, num_steps=10):
        """
        x: input tensor of shape [batch, channels, height, width]
        num_steps: number of simulation time steps (simulate repeated evaluation to mimic spiking dynamics)
        """

        # Initialize hidden states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        # Record final layer
        spk_x_rec = []
        spk_y_rec = []
        mem_x_rec = []
        mem_y_rec = []

        batch_size = x.size(0)
        outputs_x = 0
        outputs_y = 0
        for step in range(num_steps):
            # Convolutional layers with spiking activations
            x1 = self.conv1(x)
            spk1, mem1 = self.lif1(self.mp1(x1), mem1)
            
            x2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(self.mp2(x2), mem2)
            
            x3 = self.conv3(spk2)
            spk3, mem3 = self.lif3(self.mp3(x3), mem3)

            
            # Flatten features
            s3_flat = spk3.view(batch_size, -1)
            # Fully connected branches for x and y
            out_x = self.fc_x(s3_flat)
            out_y = self.fc_y(s3_flat)
            outputs_x += out_x
            outputs_y += out_y
        # Average over time steps
        outputs_x = outputs_x / num_steps
        outputs_y = outputs_y / num_steps
        
        # Apply softmax to get probabilities
        # probs_x = F.softmax(outputs_x, dim=1)
        # probs_y = F.softmax(outputs_y, dim=1)
        
        return outputs_x, outputs_y#, [spk_x_rec, spk_y_rec, mem_x_rec, mem_y_rec]
    
    def start_training(self, trainloader, optimizer, device, validationloader = None, num_steps = 10,  num_epochs=20, plot = True):
        batch_size = trainloader.batch_size
        loss_function = classification_loss
        self.training_params = {
            "type": self.name,
            "batch_size": batch_size,
            "num_steps": num_steps,
            "loss_function": loss_function.__name__,
            "optimizer": optimizer,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "num_epochs": num_epochs,
            "quantization": trainloader.dataset.quantization,
            "label_quantization": trainloader.dataset.label_quantization,
            "beta": self.beta,
            "learn_threshold": self.learn_threshold,
            "image_shape": self.image_shape,
            }
        training_loop_images(self, trainloader, optimizer, device, loss_function, validationloader, num_steps, num_epochs, plot=plot)
    
    def evaluate(self, testloader, device, num_steps, print_results=False):
        return evaluate_classification_tracker(self, testloader, device, num_steps, print_results)

class SCNNImageClassWAvg(nn.Module):
    def __init__(self, image_shape, bins_factor, beta=0.8, learn_threshold = False):
        """
        input_shape: tuple (channels, height, width) of input event data.
        """
        super(SCNNImageClassWAvg, self).__init__()

        self.name = 'ImageClassWAvg'
        self.task = 'classification'
        self.beta = beta
        self.learn_threshold = learn_threshold
        self.image_shape = image_shape
        self.weighted_avg = True
        self.max_values = {
            'X': self.image_shape[2],
            'Y': self.image_shape[1],
            'Radius': MAX_VAL_R_CAM,
            'x_cam': self.image_shape[2],
            'y_cam': MAX_VAL_Y_CAM,
            'R_cam': MAX_VAL_R_CAM
        }

        # Convolutional layers (assuming 2-channel input for polarity split)
        channels, y_pixels, x_pixels = image_shape
        x_bins, y_bins = int(x_pixels*bins_factor), int(y_pixels*bins_factor)

        self.input_shape = (channels, y_pixels, x_pixels)
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=1, padding=2)   # Expected: from (2, H, W) to (16, H/2, W/2)
        self.lif1 = snn.Leaky(beta, learn_threshold=learn_threshold)
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Expected: (32, H/4, W/4)
        self.lif2 = snn.Leaky(beta, learn_threshold=learn_threshold)
        self.mp2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Expected: (64, H/8, W/8)
        self.lif3 = snn.Leaky(beta, learn_threshold=learn_threshold)
        self.mp3 = nn.MaxPool2d(2)

        # Dynamically compute the flattened feature size by passing a dummy input.
        self.flattened_size = self._get_flattened_size(self.input_shape)
        print(f"Flattened feature size: {self.flattened_size}")
        
        # Two fully connected branches: one for x-coordinate, one for y-coordinate
        self.fc_x = nn.Linear(self.flattened_size, x_bins)
        self.lif_x = snn.Leaky(beta, learn_threshold=learn_threshold)
        self.fc_y = nn.Linear(self.flattened_size, y_bins)
        self.lif_y = snn.Leaky(beta, learn_threshold=learn_threshold)
        print(f"Number of x and y bins: {x_bins}, {y_bins}")
        
        # Surrogate gradient activation (spiking behavior simulation) IDK WHAT THIS IS
        # self.spike_fn = surrogate.fast_sigmoid(slope=25)

    def _get_flattened_size(self, input_shape):
        # Create a dummy input tensor with the given shape.
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            x = self.conv1(x)
            x = self.mp1(x)
            x = self.conv2(x)
            x = self.mp2(x)
            x = self.conv3(x)
            x = self.mp3(x)
            flattened_size = x.view(1, -1).size(1)
        return flattened_size

    def forward(self, x, num_steps=10):
        """
        x: input tensor of shape [batch, channels, height, width]
        num_steps: number of simulation time steps (simulate repeated evaluation to mimic spiking dynamics)
        """

        # Initialize hidden states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem_x = self.lif_x.init_leaky()
        mem_y = self.lif_y.init_leaky()

        # Record final layer
        spk_x_rec = []
        spk_y_rec = []
        mem_x_rec = []
        mem_y_rec = []

        batch_size = x.size(0)
        outputs_x = 0
        outputs_y = 0
        for step in range(num_steps):
            # Convolutional layers with spiking activations
            x1 = self.conv1(x)
            spk1, mem1 = self.lif1(self.mp1(x1), mem1)
            
            x2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(self.mp2(x2), mem2)
            
            x3 = self.conv3(spk2)
            spk3, mem3 = self.lif3(self.mp3(x3), mem3)

            
            # Flatten features
            s3_flat = spk3.view(batch_size, -1)
            # Fully connected branches for x and y
            out_x = self.fc_x(s3_flat)
            spk_x, mem_x = self.lif_x(out_x, mem_x)
            out_y = self.fc_y(s3_flat)
            spk_y, mem_y = self.lif_y(out_y, mem_y)
            # Record spiking activity and membrane potential for the final layer
            spk_x_rec.append(spk_x)
            spk_y_rec.append(spk_y)
            mem_x_rec.append(mem_x)
            mem_y_rec.append(mem_y)
            outputs_x += out_x
            outputs_y += out_y
        # Average over time steps
        logits_x = outputs_x / num_steps
        logits_y = outputs_y / num_steps

        # Apply softmax to get probabilities
        probs_x = torch.softmax(logits_x, dim=1)  # shape: [1, time, num_bins]
        probs_y = torch.softmax(logits_y, dim=1)

        # Compute the weighted sum (expected value):
        # Create position tensors for x and y
        positions_x = torch.linspace(0, 1, steps = probs_x.shape[1], device=probs_x.device).unsqueeze(0)
        positions_y = torch.linspace(0, 1, steps = probs_y.shape[1], device=probs_y.device).unsqueeze(0)
        
        # Adjust for batch size
        positions_x = positions_x.expand(batch_size, -1)
        positions_y = positions_y.expand(batch_size, -1)

        # Multiply each prob dist with corresponding position and sum
        weighted_x = (probs_x * positions_x).sum(dim=1).unsqueeze(1)  # shape: [batch_size, 1]
        weighted_y = (probs_y * positions_y).sum(dim=1).unsqueeze(1)  # shape: [batch_size, 1]
        
        # Apply softmax to get probabilities
        # probs_x = F.softmax(outputs_x, dim=1)
        # probs_y = F.softmax(outputs_y, dim=1)
        
        return weighted_x, weighted_y#, [spk_x_rec, spk_y_rec, mem_x_rec, mem_y_rec]
    
    def start_training(self, trainloader, optimizer, device, validationloader = None, num_steps = 10, num_epochs=20, plot = True):
        batch_size = trainloader.batch_size
        loss_function = regression_loss
        self.training_params = {
            "type": self.name,
            "batch_size": batch_size,
            "num_steps": num_steps,
            "loss_function": loss_function.__name__,
            "optimizer": optimizer,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "num_epochs": num_epochs,
            "quantization": trainloader.dataset.quantization,
            "label_quantization": trainloader.dataset.label_quantization,
            "beta": self.beta,
            "learn_threshold": self.learn_threshold,
            "image_shape": self.image_shape,
            }
        training_loop_images(self, trainloader, optimizer, device, loss_function, validationloader, num_steps, num_epochs, plot=plot)
    
    def evaluate(self, testloader, device, num_steps, print_results=False):
        return evaluate_regression_tracker(self, testloader, device, num_steps, print_results)

class SCNN3LImageRegression(nn.Module):
    def __init__(self, image_shape, beta=0.8, learn_threshold = False):
        """
        input_shape: tuple (channels, height, width) of input event data.
        """
        super(SCNN3LImageRegression, self).__init__()

        self.name = 'ImageRegression'
        self.task = 'regression'
        self.beta = beta
        self.learn_threshold = learn_threshold
        self.image_shape = image_shape
        self.max_values = {
            'X': self.image_shape[2],
            'Y': self.image_shape[1],
            'Radius': MAX_VAL_R_CAM,
            'x_cam': self.image_shape[2],
            'y_cam': MAX_VAL_Y_CAM,
            'R_cam': MAX_VAL_R_CAM
        }

        # Convolutional layers (assuming 2-channel input for polarity split)
        channels, y_pixels, x_pixels = image_shape
        self.input_shape = (channels, y_pixels, x_pixels)
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=1, padding=2)   # Expected: from (2, H, W) to (16, H/2, W/2)
        self.lif1 = snn.Leaky(beta, learn_threshold=learn_threshold)
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Expected: (32, H/4, W/4)
        self.lif2 = snn.Leaky(beta, learn_threshold=learn_threshold)
        self.mp2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Expected: (64, H/8, W/8)
        self.lif3 = snn.Leaky(beta, learn_threshold=learn_threshold)
        self.mp3 = nn.MaxPool2d(2)

        # Dynamically compute the flattened feature size by passing a dummy input.
        self.flattened_size = self._get_flattened_size(self.input_shape)
        print(f"Flattened feature size: {self.flattened_size}")
        
        # Two fully connected branches: one for x-coordinate, one for y-coordinate
        self.fc_x = nn.Linear(self.flattened_size, 1)
        self.lif_x = snn.Leaky(beta, learn_threshold=learn_threshold)
        self.fc_y = nn.Linear(self.flattened_size, 1)
        self.lif_y = snn.Leaky(beta, learn_threshold=learn_threshold)
        
        # Surrogate gradient activation (spiking behavior simulation) IDK WHAT THIS IS
        # self.spike_fn = surrogate.fast_sigmoid(slope=25)

    def _get_flattened_size(self, input_shape):
        # Create a dummy input tensor with the given shape.
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            x = self.conv1(x)
            x = self.mp1(x)
            x = self.conv2(x)
            x = self.mp2(x)
            x = self.conv3(x)
            x = self.mp3(x)
            flattened_size = x.view(1, -1).size(1)
        return flattened_size

    def forward(self, x, num_steps=10):
        """
        x: input tensor of shape [batch, channels, height, width]
        num_steps: number of simulation time steps (simulate repeated evaluation to mimic spiking dynamics)
        """

        # Initialize hidden states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem_x = self.lif_x.init_leaky()
        mem_y = self.lif_y.init_leaky()

        # Record final layer
        spk_x_rec = []
        spk_y_rec = []
        mem_x_rec = []
        mem_y_rec = []

        batch_size = x.size(0)
        outputs_x = 0
        outputs_y = 0
        for step in range(num_steps):
            # Convolutional layers with spiking activations
            x1 = self.conv1(x)
            spk1, mem1 = self.lif1(self.mp1(x1), mem1)
            
            x2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(self.mp2(x2), mem2)
            
            x3 = self.conv3(spk2)
            spk3, mem3 = self.lif3(self.mp3(x3), mem3)

            
            # Flatten features
            s3_flat = spk3.view(batch_size, -1)
            # Fully connected branches for x and y
            out_x = self.fc_x(s3_flat)
            spk_x, mem_x = self.lif_x(out_x, mem_x)
            out_y = self.fc_y(s3_flat)
            spk_y, mem_y = self.lif_y(out_y, mem_y)
            # Record spiking activity and membrane potential for the final layer
            spk_x_rec.append(spk_x)
            spk_y_rec.append(spk_y)
            mem_x_rec.append(mem_x)
            mem_y_rec.append(mem_y)
            outputs_x += out_x
            outputs_y += out_y
        # Average over time steps
        outputs_x = outputs_x / num_steps
        outputs_y = outputs_y / num_steps
            
        return outputs_x, outputs_y#, [spk_x_rec, spk_y_rec, mem_x_rec, mem_y_rec]
    
    def start_training(self, trainloader, optimizer, device, validationloader = None, num_steps = 10, num_epochs=20, plot = True):
        loss_function = regression_loss
        self.training_params = {
            "type": self.name,
            "batch_size": trainloader.batch_size,
            "num_steps": num_steps,
            "loss_function": loss_function.__name__,
            "optimizer": optimizer,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "num_epochs": num_epochs,
            "quantization": trainloader.dataset.quantization,
            "label_quantization": trainloader.dataset.label_quantization,
            "beta": self.beta,
            "learn_threshold": self.learn_threshold,
            "image_shape": self.image_shape,
            }
        training_loop_images(self, trainloader, optimizer, device, loss_function, validationloader, num_steps, num_epochs, plot=plot)
    
    def evaluate(self, testloader, device, num_steps, print_results=False):
        return evaluate_regression_tracker(self, testloader, device, num_steps, print_results)

class SCNN_Tracker_Class_GASP(nn.Module): # Three conv layers
    def __init__(self, image_shape, bins_factor, beta=0.8, learn_threshold=False):
        """
        image_shape: (channels, height, width) for input event data
        bins_factor: factor for downsampling the final x, y bins
        """
        super(SCNN_Tracker_Class_GASP, self).__init__()

        self.name = 'TrackerClassGASP'
        self.learn_threshold = learn_threshold
        self.beta = beta
        self.image_shape = image_shape
        self.max_values = {
            'x_cam': self.image_shape[2],
            'y_cam': MAX_VAL_Y_CAM,
            'R_cam': MAX_VAL_R_CAM
        }

        # Convolutional layers (assuming 2-channel input for polarity)
        channels, y_pixels, x_pixels = image_shape
        x_bins, y_bins = int(x_pixels * bins_factor), int(y_pixels * bins_factor)

        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=1, padding=2)
        self.lif1 = snn.Leaky(beta, learn_threshold=learn_threshold)
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.lif2 = snn.Leaky(beta, learn_threshold=learn_threshold)
        self.mp2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.lif3 = snn.Leaky(beta, learn_threshold=learn_threshold)
        self.mp3 = nn.MaxPool2d(2)

        # Dynamically compute the feature size after conv/pool layers
        self.flattened_size = self._get_flattened_size(image_shape)
        print(f"Flattened feature size: {self.flattened_size}")

        # Fully connected layers for x and y predictions
        self.fc_x = nn.Linear(self.flattened_size, x_bins)
        self.lif_x = snn.Leaky(beta, learn_threshold=learn_threshold)

        self.fc_y = nn.Linear(self.flattened_size, y_bins)
        self.lif_y = snn.Leaky(beta, learn_threshold=learn_threshold)

        print(f"Number of x and y bins: {x_bins}, {y_bins}")

    def _get_flattened_size(self, input_shape):
        with torch.no_grad():
            dummy_x = torch.zeros(1, *input_shape)
            out = self.conv1(dummy_x)
            out = self.conv2(out)
            out = self.conv3(out)
            # Global average pooling across spatial dims (2, 3)
            out_gap = out.mean(dim=(2, 3))  # shape: [1, 64]
            flattened_size = out_gap.view(1, -1).size(1)
        return flattened_size

    def forward(self, x, num_steps=10):
        # Initialize hidden states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem_x = self.lif_x.init_leaky()
        mem_y = self.lif_y.init_leaky()

        batch_size = x.size(0)
        outputs_x = 0
        outputs_y = 0

        for step in range(num_steps):
            # First conv/pool
            out1 = self.conv1(x)
            spk1, mem1 = self.lif1(self.mp1(out1), mem1)

            # Second conv/pool
            out2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(self.mp2(out2), mem2)

            # Third conv/pool
            out3 = self.conv3(spk2)
            spk3, mem3 = self.lif3(self.mp3(out3), mem3)

            # Global Average Pooling (spiking style)
            # shape: [batch_size, channels=64, height, width] -> [batch_size, 64]
            gap = spk3.mean(dim=(2, 3))

            # Fully connected branches
            out_x = self.fc_x(gap)
            spk_x, mem_x = self.lif_x(out_x, mem_x)

            out_y = self.fc_y(gap)
            spk_y, mem_y = self.lif_y(out_y, mem_y)

            # Accumulate raw outputs (not the spiking outputs) for final average
            outputs_x += out_x
            outputs_y += out_y

        # Average over time steps
        outputs_x = outputs_x / num_steps
        outputs_y = outputs_y / num_steps

        # Optional: Softmax if you want classification probabilities
        # probs_x = F.softmax(outputs_x, dim=1)
        # probs_y = F.softmax(outputs_y, dim=1)

        return outputs_x, outputs_y
    
    def start_training(self, trainloader, optimizer, device, validationloader = None, num_steps = 10, num_epochs=20, plot = True):
        batch_size = trainloader.batch_size
        loss_function = classification_loss
        self.training_params = {
            "type": self.name,
            "batch_size": batch_size,
            "num_steps": num_steps,
            "loss_function": loss_function.__name__,
            "optimizer": optimizer,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "num_epochs": num_epochs,
            "quantization": trainloader.dataset.quantization,
            "label_quantization": trainloader.dataset.label_quantization,
            "beta": self.beta,
            "learn_threshold": self.learn_threshold,
            "image_shape": self.image_shape,
            }
        training_loop_images(self, trainloader, optimizer, device, loss_function, validationloader, num_steps, num_epochs, plot=plot)
    
    def evaluate(self, testloader, device, num_steps, print_results=False):
        return evaluate_video_classification_tracker(self, testloader, device, num_steps, print_results)
    
class SCNNVideoClassification(nn.Module): # Based on the SCNN_Tracker3L
    def __init__(self, trainset, beta=0.8, learn_threshold = False, weighted_avg = False, dropout = False):
        """
        input_shape: tuple (channels, height, width) of input event data.
        """
        super(SCNNVideoClassification, self).__init__()

        self.name = 'VideoClassification'
        self.task = 'classification'
        self.beta = beta
        self.learn_threshold = learn_threshold
        self.image_shape = trainset.image_shape
        self.bins_factor = trainset.quantization / trainset.label_quantization
        self.weighted_avg = weighted_avg
        self.g = 9.81  # Gravitational constant for the simulation
        self.dt = 0.01  # Time step for the simulation
        self.max_values = {
            'x_cam': int(self.image_shape[2]*self.bins_factor) + 1,
            'y_cam': MAX_VAL_Y_CAM//trainset.label_quantization + 1,
            'R_cam': MAX_VAL_R_CAM//trainset.label_quantization + 1,
            'in_fov': 1,
            'X': int(self.image_shape[2]*self.bins_factor) + 1,
            'Y': int(self.image_shape[1]*self.bins_factor) + 1,
            'Radius': MAX_VAL_R_CAM//trainset.label_quantization + 1,
        }
        self.n_bins = np.ones(len(trainset.labels), dtype=int)
        self.labels = trainset.labels

        # Convolutional layers (assuming 2-channel input for polarity split)
        channels, y_pixels, x_pixels = self.image_shape

        input_shape = (channels, y_pixels, x_pixels)
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=1, padding=2)   # Expected: from (2, H, W) to (16, H/2, W/2)
        self.lif1 = snn.Leaky(beta, learn_threshold=learn_threshold)
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Expected: (32, H/4, W/4)
        self.lif2 = snn.Leaky(beta, learn_threshold=learn_threshold)
        self.mp2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Expected: (64, H/8, W/8)
        self.lif3 = snn.Leaky(beta, learn_threshold=learn_threshold)
        self.mp3 = nn.MaxPool2d(2)

        self.has_dropout = dropout
        if self.has_dropout:
            self.dropout = nn.Dropout(p=0.5)   # drop 50% of units

        # Dynamically compute the flattened feature size by passing a dummy input.
        self.flattened_size = self._get_flattened_size(input_shape)
        print(f"Flattened feature size: {self.flattened_size}")

        self.fc_layers = nn.ModuleDict()
        self.lif_layers = nn.ModuleDict()
        
        for i, label in enumerate(trainset.labels):
            n_bins = self.max_values[label]
            self.n_bins[i] = n_bins
            self.fc_layers[label] = nn.Linear(self.flattened_size, n_bins)
            self.lif_layers[label] = snn.Leaky(beta, learn_threshold=learn_threshold)
            print(f"Number of {label} bins: {n_bins}")
            


    def _get_flattened_size(self, input_shape):
        # Create a dummy input tensor with the given shape.
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            x = self.conv1(x)
            x = self.mp1(x)
            x = self.conv2(x)
            x = self.mp2(x)
            x = self.conv3(x)
            x = self.mp3(x)
            flattened_size = x.view(1, -1).size(1)
        return flattened_size

    def forward(self, sequences_lengths, membrane_potentials, num_steps_per_image=10):
        """
        x: input tensor of shape [batch, channels, height, width]
        num_steps: number of simulation time steps (simulate repeated evaluation to mimic spiking dynamics)
        """

        padded_x, lengths = sequences_lengths
        batch_size, max_seq_len = padded_x.shape[:2]

        # # Initialize hidden states
        if len(membrane_potentials) == 0:
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem3 = self.lif3.init_leaky()
        else:
            mem1, mem2, mem3 = membrane_potentials

        # Record final layer
        # Initialize tensors to store outputs for each time step per sequence.
        outputs_seq = []
        for i in range(len(self.labels)):
            outputs_seq.append(torch.zeros(batch_size, self.n_bins[i], max_seq_len, device=padded_x.device))

        total_spikes1 = 0
        total_spikes2 = 0
        total_spikes3 = 0

        for i in range(max_seq_len):
            # Create a boolean mask for samples that have a valid image at time t
            valid_mask = (i < lengths).to(padded_x.device)
            if valid_mask.sum() == 0:
                break  # No valid data at this time step for any sequence

            # Get the images for the valid sequences at time t (shape: [valid_batch, channels, height, width])
            x_t = padded_x[valid_mask, i]

            outputs = []
            for label in self.labels:
                outputs.append(torch.zeros(x_t.size(0), self.max_values[label], device=padded_x.device))

            for step in range(num_steps_per_image):
                # Convolutional layers with spiking activations
                x1 = self.conv1(x_t)
                spk1, mem1 = self.lif1(self.mp1(x1), mem1)
                
                x2 = self.conv2(spk1)
                spk2, mem2 = self.lif2(self.mp2(x2), mem2)
                
                x3 = self.conv3(spk2)
                spk3, mem3 = self.lif3(self.mp3(x3), mem3)

                
                # Flatten features
                s3_flat = spk3.view(spk3.size(0), -1)
                if self.has_dropout: s3_flat = self.dropout(s3_flat) # Dropping out 30% of the features for regularization
                
                for j, label in enumerate(self.labels):
                    fc_out = self.fc_layers[label](s3_flat)
                    # spk_out, _ = self.lif_layers[label](fc_out)
                    outputs[j].add_(fc_out)

                total_spikes1 += spk1.sum().item()
                total_spikes2 += spk2.sum().item()
                total_spikes3 += spk3.sum().item()

                # Delete intermediate tensors
                del x1, spk1, x2, spk2, x3, spk3, s3_flat
            if padded_x.device == "cuda": torch.cuda.empty_cache()
            # Average over time steps
            for j in range(len(self.labels)):
                outputs[j] = outputs[j] / num_steps_per_image
                outputs_seq[j][valid_mask, :, i] = outputs[j]


            # print('Outputs x and valid mask', outputs_x.shape, valid_mask.shape)

        C, H, W = self.image_shape
        avg_spikes_1 = total_spikes1 / (num_steps_per_image * batch_size * max_seq_len * 16 * (H // 2) * (W // 2))
        avg_spikes_2 = total_spikes2 / (num_steps_per_image * batch_size * max_seq_len * 32 * (H // 4) * (W // 4))
        avg_spikes_3 = total_spikes3 / (num_steps_per_image * batch_size * max_seq_len * 64 * (H // 8) * (W // 8))
        # print(f"Time step {i+1}/{max_seq_len}, Avg Spikes Layer 1: {avg_spikes_1:.4f}, Layer 2: {avg_spikes_2:.4f}, Layer 3: {avg_spikes_3:.4f}")
        # print(mem3.min().item(), mem3.max().item())

        # Apply softmax to get probabilities
        # probs_x = F.softmax(outputs_x, dim=1)
        # probs_y = F.softmax(outputs_y, dim=1)
        membrane_potentials = [mem1, mem2, mem3]
        return outputs_seq, membrane_potentials #, [spk_x_rec, spk_y_rec, mem_x_rec, mem_y_rec]
    
    def start_training(self, trainloader, optimizer, device, loss_function = None, validationloader = None, num_steps = 10, num_epochs=20, scheduler = None, warmup = None, plot = True, chunk_size = CHUNK_SIZE, save = [], grad_clip = False):
        batch_size = trainloader.batch_size
        if loss_function is None:
            if self.weighted_avg:
                loss_function = regression_loss
            else: 
                loss_function = classification_loss

        self.training_params = {
            "type": self.name,
            "batch_size": batch_size,
            "num_steps": num_steps,
            "loss_function": loss_function.__name__,
            "optimizer": optimizer,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "num_epochs": 0,
            "quantization": trainloader.dataset.quantization,
            "label_quantization": trainloader.dataset.label_quantization,
            "beta": self.beta,
            "learn_threshold": self.learn_threshold,
            "image_shape": self.image_shape,
            "weighted_avg": self.weighted_avg,
            "epoch_losses": [],
            "validation_errors": [],
            "validation_losses": [],
            }
        
        training_loop_videos(self, trainloader, optimizer, device, loss_function, validationloader, num_steps, num_epochs, scheduler = scheduler, warmup=warmup, plot=plot, save=save, chunk_size=chunk_size, grad_clip=grad_clip)
    
    def evaluate(self, testloader, device, num_steps, print_results=False, operation = 'mean', weighted_avg = None, chunk_size = CHUNK_SIZE):
        if weighted_avg is None:
            weighted_avg = getattr(self, "weighted_avg", False)
        return evaluate_video_regression_tracker(self, testloader, device, num_steps, print_results, operation, weighted_avg=weighted_avg) if weighted_avg else evaluate_video_classification_tracker(self, testloader, device, num_steps, print_results, operation, chunk_size=chunk_size)


class SCNNVideoRegression(nn.Module): # Based on the SCNN_Tracker3L
    def __init__(self, trainset, beta=0.8, learn_threshold = False, weighted_avg = False):
        """
        input_shape: tuple (channels, height, width) of input event data.
        """
        super(SCNNVideoRegression, self).__init__()

        self.name = 'VideoRegression'
        self.task = 'regression'
        self.beta = beta
        self.learn_threshold = learn_threshold
        self.image_shape = trainset.image_shape
        self.bins_factor = trainset.quantization / trainset.label_quantization
        self.weighted_avg = weighted_avg
        self.g = 9.81  # Gravitational acceleration in m/s^2
        self.dt = 0.01  # Time step in seconds (adjust as needed)
        self.max_values = {
            'x_cam': int(self.image_shape[2]*self.bins_factor),
            'y_cam': MAX_VAL_Y_CAM//trainset.label_quantization,
            'R_cam': MAX_VAL_R_CAM//trainset.label_quantization,
            'in_fov': 1,
        }

        # Convolutional layers (assuming 2-channel input for polarity split)
        channels, y_pixels, x_pixels = self.image_shape

        input_shape = (channels, y_pixels, x_pixels)
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=1, padding=2)   # Expected: from (2, H, W) to (16, H/2, W/2)
        self.lif1 = snn.Leaky(beta, learn_threshold=learn_threshold)
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Expected: (32, H/4, W/4)
        self.lif2 = snn.Leaky(beta, learn_threshold=learn_threshold)
        self.mp2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Expected: (64, H/8, W/8)
        self.lif3 = snn.Leaky(beta, learn_threshold=learn_threshold)
        self.mp3 = nn.MaxPool2d(2)

        # Dynamically compute the flattened feature size by passing a dummy input.
        self.flattened_size = self._get_flattened_size(input_shape)
        print(f"Flattened feature size: {self.flattened_size}")
        
        # Two fully connected branches: one for x-coordinate, one for y-coordinate
        self.fc_x = nn.Linear(self.flattened_size, 1)
        self.fc_y = nn.Linear(self.flattened_size, 1)
        self.fc_z = nn.Linear(self.flattened_size, 1)


    def _get_flattened_size(self, input_shape):
        # Create a dummy input tensor with the given shape.
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            x = self.conv1(x)
            x = self.mp1(x)
            x = self.conv2(x)
            x = self.mp2(x)
            x = self.conv3(x)
            x = self.mp3(x)
            flattened_size = x.view(1, -1).size(1)
        return flattened_size

    def forward(self, sequences_lengths, num_steps_per_image=10):
        """
        x: input tensor of shape [batch, channels, height, width]
        num_steps: number of simulation time steps (simulate repeated evaluation to mimic spiking dynamics)
        """

        padded_x, lengths = sequences_lengths
        batch_size, max_seq_len = padded_x.shape[:2]

        # # Initialize hidden states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        # Record final layer
        # Initialize tensors to store outputs for each time step per sequence.
        outputs_seq_x = torch.zeros(batch_size, self.fc_x.out_features, max_seq_len, device=padded_x.device)
        outputs_seq_y = torch.zeros(batch_size, self.fc_y.out_features, max_seq_len, device=padded_x.device)
        outputs_seq_z = torch.zeros(batch_size, self.fc_z.out_features, max_seq_len, device=padded_x.device)

        for i in range(max_seq_len):
            # Create a boolean mask for samples that have a valid image at time t
            # if len(lengths) == 1:
            #     valid_mask = torch.tensor([True])
            # else:
            valid_mask = (i < lengths).to(padded_x.device)
            if valid_mask.sum() == 0:
                break  # No valid data at this time step for any sequence

            # Get the images for the valid sequences at time t (shape: [valid_batch, channels, height, width])
            x_t = padded_x[valid_mask, i]

            total_spikes1 = 0
            total_spikes2 = 0
            total_spikes3 = 0

            outputs_x = 0
            outputs_y = 0
            outputs_z = 0
            for step in range(num_steps_per_image):
                # Convolutional layers with spiking activations
                x1 = self.conv1(x_t)
                spk1, mem1 = self.lif1(self.mp1(x1), mem1)
                
                x2 = self.conv2(spk1)
                spk2, mem2 = self.lif2(self.mp2(x2), mem2)
                
                x3 = self.conv3(spk2)
                spk3, mem3 = self.lif3(self.mp3(x3), mem3)

                
                # Flatten features
                s3_flat = spk3.view(spk3.size(0), -1)
                # Fully connected branches for x and y
                out_x = torch.sigmoid(self.fc_x(s3_flat))
                out_y = torch.sigmoid(self.fc_y(s3_flat))
                out_z = torch.sigmoid(self.fc_z(s3_flat))

                outputs_x += out_x
                outputs_y += out_y
                outputs_z += out_z

                total_spikes1 += spk1.sum(dim=(1, 2, 3))
                total_spikes2 += spk2.sum(dim=(1, 2, 3))
                total_spikes3 += spk3.sum(dim=(1, 2, 3))

                # Delete intermediate tensors
                del x1, spk1, x2, spk2, x3, spk3, s3_flat, out_x, out_y
            if padded_x.device == "cuda": torch.cuda.empty_cache()
            # Average over time steps
            outputs_x = outputs_x / num_steps_per_image
            outputs_y = outputs_y / num_steps_per_image
            outputs_z = outputs_z / num_steps_per_image

            # print('Outputs x and valid mask', outputs_x.shape, valid_mask.shape)

            outputs_seq_x[valid_mask, :, i] = outputs_x
            outputs_seq_y[valid_mask, :, i] = outputs_y
            outputs_seq_z[valid_mask, :, i] = outputs_z

            # Print average spikes per layer
        C, H, W = self.image_shape
        avg_spikes_1 = total_spikes1 / (num_steps_per_image * batch_size * max_seq_len * 16 * (H // 2) * (W // 2))
        avg_spikes_2 = total_spikes2 / (num_steps_per_image * batch_size * max_seq_len * 32 * (H // 4) * (W // 4))
        avg_spikes_3 = total_spikes3 / (num_steps_per_image * batch_size * max_seq_len * 64 * (H // 8) * (W // 8))
        print(f"Time step {i+1}/{max_seq_len}, Avg Spikes Layer 1: {avg_spikes_1:.4f}, Layer 2: {avg_spikes_2:.4f}, Layer 3: {avg_spikes_3:.4f}")
        
        
        return outputs_seq_x, outputs_seq_y, outputs_seq_z
    
    def start_training(self, trainloader, optimizer, device, loss_function, validationloader = None, num_steps = 10, num_epochs=20, plot = True, chunk_size=CHUNK_SIZE, save = []):
        batch_size = trainloader.batch_size

        self.training_params = {
            "type": self.name,
            "batch_size": batch_size,
            "num_steps": num_steps,
            "loss_function": loss_function.__name__,
            "optimizer": optimizer,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "num_epochs": 0,
            "quantization": trainloader.dataset.quantization,
            "label_quantization": trainloader.dataset.label_quantization,
            "beta": self.beta,
            "learn_threshold": self.learn_threshold,
            "image_shape": self.image_shape,
            "weighted_avg": self.weighted_avg,
            }
        
        training_loop_videos(self, trainloader, optimizer, device, loss_function, validationloader, num_steps, num_epochs, plot=plot, chunk_size=chunk_size, save=save)
    
    def evaluate(self, testloader, device, num_steps, print_results=False, operation = 'mean', chunk_size=CHUNK_SIZE):
        return evaluate_video_regression_tracker(self, testloader, device, num_steps, print_results, operation, chunk_size=chunk_size)


def training_loop_images(model, trainloader, optimizer, device, loss_function, validationloader = None, num_steps = 10, num_epochs=20, plot = True):
    epoch_losses = []
    validation_errors = []
    if model.max_values:
        max_values = [model.max_values[label] for label in trainloader.dataset.labels]
    else:
        max_values = None
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (images, labels) in enumerate(trainloader):
            images = images.to(device)  # shape: [batch, channels, height, width]
            # Use only x and y coordinates for classification.
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images, num_steps=num_steps)
            loss = loss_function(model, logits, labels, max_values=max_values)
            
            loss.backward()
            if hasattr(model, "grad_clipping") and model.grad_clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(trainloader)
        epoch_losses.append(avg_loss)
        if validationloader is not None:
            error = model.evaluate(validationloader, device, num_steps, print_results=False)
            error = np.linalg.norm(error).item()
            validation_errors.append(error)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Validation Error: {error:.4f} pixels")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        if device == "cuda":
            torch.cuda.empty_cache()      
    
    # Plot loss vs. epochs
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, num_epochs+1), epoch_losses, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.title("Training Loss vs. Epochs")
        plt.grid(True)
        if validationloader is not None:
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, num_epochs+1), validation_errors, marker='o')
            plt.xlabel("Epoch")
            plt.ylabel("Validation Error")
            plt.title("Validation Error vs. Epochs")
            plt.grid(True)
        plt.show()


def training_loop_videos(model, trainloader, optimizer, device, loss_function, validationloader = None, num_steps = 10, num_epochs=20, scheduler = None, plot = True,
                         warmup = None, chunk_size=CHUNK_SIZE, save = [], grad_clip = False):
    epoch_losses = []
    validation_errors = []
    max_values = [model.max_values[label] for label in trainloader.dataset.labels]
    best_val = float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for padded_imgs, padded_labels, lengths in trainloader:
            # Move everything to device once
            padded_imgs   = padded_imgs.to(device)   # [B, T, C, H, W]
            padded_labels = padded_labels.to(device) # [B, T, …]
            lengths       = lengths.to(device)       # [B]

            optimizer.zero_grad()
            total_chunks = 0

            # iterate over each chunk of the sequence
            T = padded_imgs.size(1)
            total_chunks = T // chunk_size + (T % chunk_size > 0)
            membrane_potentials = []
            for t0 in range(0, T, chunk_size):
                t1 = min(t0 + chunk_size, T)
                imgs_chunk   = padded_imgs[:, t0:t1]   # [B, chunk, C, H, W]
                labels_chunk = padded_labels[:, t0:t1] # [B, chunk, …]
                # adjust lengths for this chunk
                # mask out frames ≥ original length
                frame_idx = torch.arange(t0, t1, device=device).unsqueeze(0)  # [1, chunk]
                valid_mask = frame_idx < lengths.unsqueeze(1)                  # [B, chunk]

                # forward pass on just this chunk
                logits, new_mem = model((imgs_chunk, lengths), membrane_potentials, num_steps_per_image=num_steps)
                membrane_potentials = [m.detach() for m in new_mem]

                # print(labels_chunk.shape)
                # outputs_array = [labels_chunk[:,:,i] for i in range(3)]

                loss  = loss_function(model, logits, labels_chunk, mask=valid_mask, max_values=max_values)


                # Accumulate grads over chunks and step :
                loss.backward()
                # max_grad_norm = 0.0
                epoch_loss += loss.item()
                if grad_clip: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # for p in model.parameters():
                #     if p.grad is not None:
                #         param_norm = p.grad.data.norm(2)  # L2 norm
                #         max_grad_norm = max(max_grad_norm, param_norm.item())
                # print(f"Max gradient norm for epoch {epoch+1}, chunk {total_chunks}: {max_grad_norm:.4f}")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if warmup is not None and epoch == 0:
                    warmup.step()
                total_chunks += 1

                # free up memory
                del imgs_chunk, labels_chunk, logits, loss, valid_mask

            # Step here:
            torch.cuda.empty_cache()
        avg_loss = epoch_loss / trainloader.dataset.return_n_frames()
        model.training_params["epoch_losses"].append(avg_loss)

        if validationloader is not None:
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for padded_imgs, padded_labels, lengths in validationloader:
                    # Move everything to device once
                    padded_imgs   = padded_imgs.to(device)   # [B, T, C, H, W]
                    padded_labels = padded_labels.to(device) # [B, T, …]
                    lengths       = lengths.to(device)       # [B]

                    total_chunks = 0

                    # iterate over each chunk of the sequence
                    T = padded_imgs.size(1)
                    total_chunks = T // chunk_size + (T % chunk_size > 0)
                    membrane_potentials = []
                    print_results_flag = False # Set this to False to avoid printing results
                    for t0 in range(0, T, chunk_size):
                        t1 = min(t0 + chunk_size, T)
                        imgs_chunk   = padded_imgs[:, t0:t1]   # [B, chunk, C, H, W]
                        labels_chunk = padded_labels[:, t0:t1] # [B, chunk, …]
                        # adjust lengths for this chunk
                        # mask out frames ≥ original length
                        frame_idx = torch.arange(t0, t1, device=device).unsqueeze(0)  # [1, chunk]
                        valid_mask = frame_idx < lengths.unsqueeze(1)                  # [B, chunk]

                        # forward pass on just this chunk
                        logits, new_mem = model((imgs_chunk, lengths), membrane_potentials, num_steps_per_image=num_steps)
                        membrane_potentials = [m.detach() for m in new_mem]

                        loss  = loss_function(model, logits, labels_chunk, mask=valid_mask, max_values=max_values)
                        print_results_flag = False # Only print results for the first chunk

                        val_loss += loss.item()
                        if grad_clip: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                        if warmup is not None and epoch == 0:
                            warmup.step()
                        total_chunks += 1

                        # free up memory
                        del imgs_chunk, labels_chunk, logits, loss, valid_mask

                    # Step here:
                    torch.cuda.empty_cache()
                avg_val_loss = val_loss / validationloader.dataset.return_n_frames()
                model.training_params["validation_losses"].append(avg_val_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f} pixels")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        model.training_params["num_epochs"] += 1
        
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            epochs_no_improve = 0
            save_model(model, 'models/best_model_current_training.pt')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= 25:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if scheduler is not None:
            try:
                scheduler.step(avg_val_loss)
            except:
                raise ValueError("Scheduler step failed. Ensure the scheduler is compatible with the training loop (ReduceLROnPlateau).")
            print(f"Learning rate after epoch {epoch+1}: {scheduler.get_last_lr()[0]:.6f}")


        if epoch + 1 in save:
            save_model(model)

        # Plot loss vs. epochs
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, epoch+2), model.training_params["epoch_losses"], marker='o', label = 'Taining Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.title("Loss vs. Epochs")
        plt.grid(True)
        if validationloader is not None:
            plt.plot(range(1, epoch+2), model.training_params["validation_losses"], marker='o', label = 'Validation Loss')
            plt.legend()
        plt.show()


########################################################################
#                       LOSS FUNCTIONS                                 #
########################################################################        

def ordinal_loss_criterion(logits, y):
    # logits: [B, K-1] ;  y: [B] integer classes
    K_minus_1 = logits.size(1)
    # build target matrix t(y) in one line
    targets = (torch.arange(K_minus_1, device=y.device)[None, :] < y[:, None]).float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return bce.sum(dim=1).mean()   


def classification_loss(model, outputs, labels, max_values = None, mask = None, print_results = False): # Max_values is not used here, but it is used in the regression loss
    # criterion = nn.CrossEntropyLoss()
    criterion = ordinal_loss_criterion if hasattr(model, "ordinal") and model.ordinal else nn.CrossEntropyLoss(label_smoothing=0.3) # label_smoothing is used to avoid overfitting to the training set
    total_loss = 0
    if mask == None:
        # loss_x = criterion(logits_x, labels_x.round().long())
        # loss_y = criterion(logits_y, labels_y.round().long())
        for i, logit in enumerate(outputs):
            loss = criterion(logit, labels[:,i].round().long())
            total_loss += loss
        del loss, logit
    else:
        # batch, num_classes_x, max_seq_length = logits[0].shape
        # Create a mask: for each sequence in the batch, valid if t < length
        # The mask will have shape [batch, max_seq_length]
        # mask = torch.arange(max_seq_length, device=logits[0].device).expand(batch, max_seq_length) < lengths.unsqueeze(1)              # [batch * max_seq_length]
        for i, logit in enumerate(outputs):
            labels_valid = labels[:, :, i][mask]
            # print(labels_valid.max(), labels_valid.min())
            logit = logit.transpose(2, 1)
            logit_valid = logit[mask]
            # print(logit_valid.shape)
            loss = criterion(logit_valid, labels_valid.round().long())
            if print_results: 
                print(f"Loss for {model.labels[i]}: {loss.item()}")
            total_loss += loss
        del logit, labels_valid, logit_valid, loss
    return total_loss

def regression_loss(model, outputs, labels, max_values, mask = None):
    criterion = nn.MSELoss()
    total_loss = 0
    if hasattr(model, "weighted_avg") and model.weighted_avg:
        outputs = logits_to_weighted_avg(outputs)
    if mask == None:
        for i, (output, max_value) in enumerate(zip(outputs, max_values)):
            pred = output * max_value
            loss = criterion(pred, labels[:,i].unsqueeze(1))
            total_loss += loss
        del loss, output, pred, max_value
    else: # I am not sure if this version for videos works, but I dont have a regression model for videos yet
        # batch, num_classes_x, max_seq_length = outputs[0].shape
        # Create a mask: for each sequence in the batch, valid if t < length
        # The mask will have shape [batch, max_seq_length]
        # mask = torch.arange(max_seq_length, device=outputs[0].device).expand(batch, max_seq_length) < lengths.unsqueeze(1)              # [batch * max_seq_length]
        for i, (output, max_value) in enumerate(zip(outputs, max_values)):
            labels_valid = labels[:, :, i][mask]
            output = output.transpose(2, 1)
            output_valid = output[mask] * max_value
            loss = criterion(output_valid.squeeze(-1), labels_valid)
            total_loss += loss
        del output, labels_valid, output_valid, loss, max_value
    return total_loss

def pinn_loss(model,
              outputs,             # what your model returns for this chunk
              labels,             # labels_chunk: shape [B, chunk_len, 3]
              mask,               # valid_mask: shape [B, chunk_len]
              max_values=None):

    """
    Physics‐Informed 3D loss for one chunk, given:
      - model: PINN with .dt, .g, .mu
      - logits: tuple (preds_x, preds_y, preds_z), each [B, chunk_len]
      - labels: tensor [B, chunk_len, 3]  (columns: x,y,z)
      - mask:   boolean tensor [B, chunk_len] marking real frames
      - max_values: max values for each dimension
    """
    if model.task == 'regression':
        preds = [(o * m).squeeze() for o, m in zip(outputs, max_values)]
    elif model.task == 'classification':
        if getattr(model, "weighted_avg", False):
            preds = logits_to_weighted_avg(outputs)
        else:
            preds = outputs
    else:
        raise ValueError("Invalid task type. Expected 'regression' or 'classification'.")
    # print('Preds', preds[0].shape)
    B, chunk_len = preds[0].shape

    # This is supposed to be the same as below. This is left for readability purposes.
    preds_valid = [p[mask] for p in preds]    # list of D tensors, each [N_valid]
    labels_valid = labels[mask]               # [N_valid, D]

    # 2) Compute squared errors per dimension
    sq_errs = [
        (preds_valid[i] - labels_valid[:, i])**2
        for i in range(len(preds_valid))
    ]  # list of D tensors, each [N_valid]

    # 3) Sum across dims → [N_valid], then mean over all samples
    err_sum = torch.stack(sq_errs, dim=0).sum(dim=0)  # [N_valid]
    print(err_sum.shape)
    L_data  = err_sum.mean()                         # scalar

    # L_data = torch.mean(
    #     sum((p[mask] - labels[mask, i])**2
    #         for i, p in enumerate(preds))
    # )
    # 2) Physics loss: we need at least 3 frames, so we'll only compute
    #    over those indices i where i+2 < original sequence length. But
    #    since this is a chunk, we mask again for positions >= 2.
    #    We'll reuse `mask` shifted by 2 steps.
    if chunk_len < 3:
        # too short for any physics residual
        return L_data

    # Stack into (B, 3, chunk_len)
    pos = CAMERA.project_ball_camera_to_world(preds, True)
    # print('world_coords', world_coords.shape)
    # pos = torch.stack([*world_coords], dim=1)
    dt = model.dt

    # second-order central difference (B,chunk_len-2,3)
    dd = (pos[:, 2:, :] - 2*pos[:, 1:-1, :] + pos[:, :-2, :]) / (dt**2)
    # first-order central diff for velocity (B,chunk_len-2,3)
    dv = (pos[:, 2:, :] - pos[:, :-2, :]) / (2*dt)

    # build physics mask: only those frames in the chunk where the
    # original mask was True at i, i+1, and i+2.
    # mask[:, :-2] marks frames where i < length-2
    phys_mask = mask[:, :-2] & mask[:, 1:-1] & mask[:, 2:]

    # flatten residuals
    dd_x = dd[:, :, 0][phys_mask]
    dd_y = dd[:, :, 1][phys_mask]
    dd_z = dd[:, :, 2][phys_mask]
    # dv_x = dv[:, 0, :][phys_mask]
    # dv_y = dv[:, 1, :][phys_mask]
    dv_z = dv[:, :, 2][phys_mask]

    # soft‐switch weight based on height at the middle frame
    z_mid = pos[:, 1:-1, 2][phys_mask]  # (sum(phys_mask),)

    eps, k = 0.22, 100.0
    # s_f = torch.sigmoid(k * (z_mid - BALL_RADIUS - eps))
    s_f = 1
    # s_r = 1 - torch.sigmoid(k * (z_mid - BALL_RADIUS - 0.01))
    s_r = 0
    print('Warning: only flying loss')

    # flight residual
    r_f = dd_x**2 + dd_y**2 + (dd_z + model.g)**2

    # roll residual
    contact    = (z_mid-BALL_RADIUS)**2 + dv_z**2
    # friction_x = (dv_x + model.mu * model.g * torch.sign(dv_x))**2
    # friction_y = (dv_y + model.mu * model.g * torch.sign(dv_y))**2
    r_r        = contact #+ friction_x + friction_y

    L_phys = torch.mean(s_f * r_f + s_r * r_r)
    C = 0.1
    print('L_phys', C*L_phys.item(), 'L_data', L_data.item())

    return L_data + C * L_phys

def classification_loss_w_confidence(model, outputs, labels, max_values = None, mask = None, gate_by_pred = False, alpha = 100, print_results = False): # Max_values is not used here, but it is used in the regression loss
    criterion_class = nn.CrossEntropyLoss()
    criterion_regression = nn.BCEWithLogitsLoss()
    conf_labels = labels[:, :, -1]
    conf_outputs = outputs[-1].squeeze(1)
    conf_outputs_sig = torch.sigmoid(conf_outputs)
    outputs = outputs[:-1]
    labels = labels[:, :, :-1]
    cls_loss = 0
    if mask is None:
        loss_conf = criterion_regression(conf_outputs, conf_labels)
        for i, logit in enumerate(outputs):
            loss = criterion_class(logit, labels[:,i].round().long())
            cls_loss += loss
        total_loss = alpha * loss_conf + conf_outputs_sig * cls_loss.mean()
    else:
        conf_labels_valid = conf_labels[mask]
        conf_outputs_valid = conf_outputs[mask]
        loss_conf = criterion_regression(conf_outputs_valid, conf_labels_valid)
        gate = conf_outputs_sig[mask] if gate_by_pred else conf_labels[mask]  # (B,)
        for i, logit in enumerate(outputs):
            labels_valid = labels[mask][:, i].round().long()
            logit = logit.transpose(2, 1)
            logit_valid = logit[mask]
            loss = F.cross_entropy(
                logit_valid,
                labels_valid,
                reduction='none'
            )
            loss = gate * loss  # Apply gate to the loss
            cls_loss += loss
            if print_results:
                print(f"Loss for {model.labels[i]}: {loss.mean().item()}")
        cls_loss = cls_loss.mean()
        loss_conf = alpha * loss_conf
        if print_results:
            print('Loss conf', loss_conf.item(), 'cls_loss', cls_loss.item())
        # print('Loss conf', loss_conf.item(), 'cls_loss', cls_loss.item())
        total_loss = loss_conf + cls_loss
    del logit, labels_valid, logit_valid, loss_conf, cls_loss, conf_labels_valid, conf_outputs_valid, gate, loss, conf_labels, conf_outputs
    return total_loss

def classification_loss_without_confidence(model, outputs, labels, max_values = None, mask = None, gate_by_pred = False, alpha = 100): # Max_values is not used here, but it is used in the regression loss
    criterion_class = nn.CrossEntropyLoss()
    conf_labels = labels[:, :, -1]
    outputs = outputs[:-1]
    labels = labels[:, :, :-1]
    cls_loss = 0
    if mask is None:
        for i, logit in enumerate(outputs):
            loss = criterion_class(logit, labels[:,i].round().long())
            cls_loss += loss
        total_loss = cls_loss.mean()
    else:
        gate = conf_labels[mask]  # (B,)
        for i, logit in enumerate(outputs):
            labels_valid = labels[mask][:, i].round().long()
            logit = logit.transpose(2, 1)
            logit_valid = logit[mask]
            loss = criterion_class(logit_valid, labels_valid.round().long())
            # print('Loss', loss.mean(), gate.mean())
            cls_loss += gate * loss
        cls_loss = cls_loss.mean()
        # print('Loss conf', loss_conf.item(), 'cls_loss', cls_loss.item())
        total_loss = cls_loss
    del logit, labels_valid, logit_valid, cls_loss, gate, loss, conf_labels
    return total_loss

def classification_loss_just_confidence(model, outputs, labels, max_values = None, mask = None, gate_by_pred = False): # Max_values is not used here, but it is used in the regression loss
    criterion_regression = nn.BCEWithLogitsLoss()
    conf_labels = labels[:, :, -1]
    conf_outputs = outputs[-1].squeeze(1)
    if mask is None:
        loss_conf = criterion_regression(conf_outputs, conf_labels)
        total_loss = loss_conf
    else:
        conf_labels_valid = conf_labels[mask]
        conf_outputs_valid = conf_outputs[mask]
        loss_conf = criterion_regression(conf_outputs_valid, conf_labels_valid)
        # print('Loss conf', loss_conf.item(), 'cls_loss', cls_loss.item())
        total_loss = loss_conf
    del loss_conf,conf_labels_valid, conf_outputs_valid, conf_labels, conf_outputs
    return total_loss


def logits_to_weighted_avg(logits):
    weighted_vars = []
    batch_size = logits[0].shape[0]
    for logit in logits:
        probs = torch.softmax(logit, dim=1)  

        # Compute the weighted sum (expected value):
        # Create position tensors for x and y
        values = torch.linspace(0, 1, steps = probs.shape[1], device=probs.device).unsqueeze(0)

        # Adjust for batch size
        values = values.expand(batch_size, -1).unsqueeze(2)  # shape: [batch_size, x_bins, 1]

        # Multiply each prob dist with corresponding position and sum
        weighted_var = (probs * values).sum(dim=1)  # shape: [batch_size, time]
        weighted_vars.append(weighted_var)
    return weighted_vars

# def plot_error_distribution(errors, n_bins=10, fields = ['x', 'y', 'R']):

#     total_errors = np.linalg.norm(errors, axis=0)
    
#     # Add epsilon to avoid zero values
#     epsilon = 1e-9
#     total_errors_safe = total_errors + epsilon

#     # Create plot
#     plt.figure(figsize=(12, 6))
#     # Calculate 95% confidence intervals
#     confidence_level = 0.95
#     z_score = norm.ppf(confidence_level)  # For normal distributions

#     for i in range(len(errors)):
#         # Fit normal distributions to X/Y errors
#         mu, std = norm.fit(errors[i])
#         ci = mu + z_score * std
#         field = fields[i]
#         print(f"Error {i}: μ={mu:.2f}, σ={std:.2f}, 95% CI={ci:.3f} pixels")
#         # Plot histogram
#         plt.hist(errors[i], bins=n_bins, alpha=0.5, density=True, label=f'Error {field}', color='C'+str(i))
#         # Generate x values for the normal distribution
#         # x = np.linspace(min(errors[i]), max(errors[i]), 1000)
#         x = np.linspace(min(errors[i]), 60, 1000)
#         # Plot the normal distribution
#         plt.plot(x, norm.pdf(x, mu, std), 'r--', lw=2, label=f'{field} fit: μ={mu:.2f}, σ={std:.2f}')
#         # Plot 95% CI lines
#         plt.axvline(ci, color='red', linestyle=':', linewidth=2, label=f'{field} 95% CI: {ci:.3f}')


#     # Fit Rayleigh distribution to total error
#     # rayleigh_params = rayleigh.fit(total_errors_safe, floc=0)  # Fix location at 0
#     # scale_total = rayleigh_params[1]  # Scale parameter (σ) of Rayleigh distribution
#     # ci_total = rayleigh.ppf(confidence_level, scale=scale_total)  # Rayleigh CI

#     # print(f"Total errors: σ={scale_total:.2f}, 95% CI={ci_total:.3f} pixels")

#     # plt.hist(total_errors_safe, bins=n_bins, alpha=0.5, density=True, label='Total errors', color='blue')

#     # x_total = np.linspace(0, max(total_errors_safe), 1000)  # Rayleigh starts at 0
#     # plt.plot(x_total, rayleigh.pdf(x_total, scale=scale_total), 'b--', lw=2, 
#     #         label=f'Total fit: σ={scale_total:.2f}')

#     # plt.axvline(ci_total, color='blue', linestyle=':', linewidth=2, 
#     #             label=f'Total 95% CI: {ci_total:.3f}')

#     plt.xlabel('Error')
#     plt.ylabel('Probability Density')
#     plt.title('Error Distribution with 95% Confidence Intervals')
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.grid(True)
#     plt.tight_layout()


def plot_error_distribution(errors, n_bins=30, fields=['x', 'y', 'R']):
    # Define bins over [-60, 60]
    bins = np.linspace(-60, 60, n_bins + 1)
    
    # Colors for each field
    colors = ['C0', 'C1', 'C2']
    
    # Setup figure
    plt.figure(figsize=(12, 6))
    confidence_level = 0.95
    z_score = norm.ppf((1 + confidence_level) / 2)   # two‐sided
    
    # X‐axis values for full Gaussian curve
    x_vals = np.linspace(-60, 60, 1000)
    
    for i, errs in enumerate(errors):
        mu, std = norm.fit(errs)
        ci_low  = mu - z_score * std
        ci_high = mu + z_score * std
        field = fields[i]
        
        print(f"Error {field}: μ={mu:.2f}, σ={std:.2f}, 95% CI=[{ci_low:.2f}, {ci_high:.2f}] pixels")
        print(f"Durbin–Watson {field}: {durbin_watson(errs):.2f}")
        
        # Histogram
        plt.hist(errs, bins=bins, alpha=0.4, density=True,
                 label=f'{field} errors', color=colors[i])
        
        # Full Gaussian PDF
        plt.plot(x_vals, norm.pdf(x_vals, mu, std),
                 linestyle='--', linewidth=2,
                 label=f'{field} fit (μ={mu:.2f}, σ={std:.2f})',
                 color=colors[i])
        
        # CI bounds
        plt.axvline(ci_low,  color=colors[i], linestyle=':', linewidth=2,
                    label=f'{field} –1σ bound')
        plt.axvline(ci_high, color=colors[i], linestyle=':', linewidth=2,
                    label=f'{field} +1σ bound')
    
#     # Fit Rayleigh distribution to total error
#     # rayleigh_params = rayleigh.fit(total_errors_safe, floc=0)  # Fix location at 0
#     # scale_total = rayleigh_params[1]  # Scale parameter (σ) of Rayleigh distribution
#     # ci_total = rayleigh.ppf(confidence_level, scale=scale_total)  # Rayleigh CI

#     # print(f"Total errors: σ={scale_total:.2f}, 95% CI={ci_total:.3f} pixels")

#     # plt.hist(total_errors_safe, bins=n_bins, alpha=0.5, density=True, label='Total errors', color='blue')

#     # x_total = np.linspace(0, max(total_errors_safe), 1000)  # Rayleigh starts at 0
#     # plt.plot(x_total, rayleigh.pdf(x_total, scale=scale_total), 'b--', lw=2, 
#     #         label=f'Total fit: σ={scale_total:.2f}')

#     # plt.axvline(ci_total, color='blue', linestyle=':', linewidth=2, 
#     #             label=f'Total 95% CI: {ci_total:.3f}')

    # Finalize plot limits and labels
    # plt.xlim( 0, 60)

    # Labels & legend
    plt.xlabel('Error (pixels)')
    plt.ylabel('Probability Density')
    plt.title('Full Gaussian Fits to Error Distributions')
    plt.xlim(-60, 60)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def evaluate_classification_tracker(model, testloader, device, num_steps=10, print_results=True, operation="mean"):
    """
    Evaluate the model on the testset.
    
    Parameters:
        testset: A dataset object.
        device: Device to run on ("cuda" or "cpu").
        num_steps: Number of simulation time steps to use in the model.
        print_results: Whether to print the results.
        
    Returns:
        avg_error_x, avg_error_y: Average absolute error for x and y predictions.
    """
    model.eval()  # Set model to evaluation mode
    
    all_errors_x = np.array([])
    all_errors_y = np.array([])

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            # Assume labels contain [x, y, ...]; round and convert to int indices
            labels_x = labels[:, 0].round().long().to(device)
            labels_y = labels[:, 1].round().long().to(device)
            
            # Get model outputs (raw logits for x and y)
            outputs_x, outputs_y = model(images, num_steps=num_steps)
            # Use argmax to get the predicted bin (x and y coordinates)
            preds_x = torch.argmax(outputs_x, dim=1)
            preds_y = torch.argmax(outputs_y, dim=1)
            
            # Compute absolute error per sample
            error_x = torch.abs(preds_x.float() - labels_x.float())
            error_y = torch.abs(preds_y.float() - labels_y.float())
            
            all_errors_x = np.concatenate((all_errors_x, error_x.cpu().numpy()))
            all_errors_y = np.concatenate((all_errors_y, error_y.cpu().numpy()))
        if operation == "mean":
            avg_error_x = np.mean(all_errors_x)
            avg_error_y = np.mean(all_errors_y)
            if print_results: print(f"Average X Error: {avg_error_x:.4f} pixels, Average Y Error: {avg_error_y:.4f} pixels")
            return avg_error_x, avg_error_y
        elif operation == "distribution":
            plot_error_distribution(all_errors_x, all_errors_y) # This doesnt work anymore. Make it work for N variables
            return all_errors_x, all_errors_y

def evaluate_video_classification_tracker(model, testloader, device, num_steps=10, print_results=True, operation="mean", chunk_size=CHUNK_SIZE):
    model.eval()  # Set model to evaluation mode
    print("Evaluating video classification tracker")
    n_fields = len(testloader.dataset.labels)
    if n_fields == 4:
        confidence_flag = True
    else:
        confidence_flag = False
    all_errors = [[] for _ in range(n_fields)]
    all_gts = [[] for _ in range(n_fields)]
    all_preds = [[] for _ in range(n_fields)]
        
    with torch.no_grad():
        for padded_imgs, padded_labels, lengths in testloader:
            # Move everything to device once
            padded_imgs   = padded_imgs.to(device)   # [B, T, C, H, W]
            padded_labels = padded_labels.round().long().to(device)
            lengths       = lengths.to(device)       # [B]
            
            # iterate over each chunk of the sequence
            T = padded_imgs.size(1)
            membrane_potentials = []
            printed = True
            for t0 in range(0, T, chunk_size):
                t1 = min(t0 + chunk_size, T)
                imgs_chunk   = padded_imgs[:, t0:t1]   # [B, chunk, C, H, W]
                labels_chunk = padded_labels[:, t0:t1] # [B, chunk, …]
                # adjust lengths for this chunk
                # mask out frames ≥ original length
                frame_idx = torch.arange(t0, t1, device=device).unsqueeze(0)  # [1, chunk]
                valid_mask = frame_idx < lengths.unsqueeze(1)                  # [B, chunk]

                # forward pass on just this chunk
                outputs, membrane_potentials = model((imgs_chunk, lengths), membrane_potentials, num_steps_per_image=num_steps)

                labels_valid = labels_chunk[valid_mask]
                if confidence_flag:
                    labels_valid, conf_labels_valid = labels_valid[:,:-1], labels_valid[:,-1]
                    outputs, output_confidence = outputs[:-1], outputs[-1].squeeze(1)
                    # print('Confidence', output_confidence.shape, 'Conf Labels', conf_labels_valid.shape)
                    # print('Outputs', outputs[0].shape, 'Labels valid', labels_valid.shape)
                for i, output in enumerate(outputs):
                    output = output.transpose(2, 1)
                    output_valid = output[valid_mask]
                    pred = torch.argmax(output_valid, dim=1)
                    if not printed:
                        print('Pred', pred, 'Labels valid', labels_valid[:, i])
                        printed = True
                    # errors = torch.abs(pred.squeeze(-1) - labels_valid[:, i])
                    errors = pred.squeeze(-1) - labels_valid[:, i]
                    # print('Errors', errors.shape)
                    all_errors[i].extend(errors.cpu().tolist())
                    all_gts[i].extend(labels_valid[:, i].cpu().tolist())
                    all_preds[i].extend(pred.cpu().tolist())
                if confidence_flag:
                    # print('Confidence', padded_confidence_valid.shape)
                    confidence_probs = torch.sigmoid(output_confidence)
                    conf_probs_valid = confidence_probs[valid_mask]
                    errors = torch.abs(conf_probs_valid - conf_labels_valid)
                    all_errors[-1].extend(errors.cpu().tolist())
                    all_gts[-1].extend(labels_valid[:, -1].cpu().tolist())
                    all_preds[-1].extend(conf_probs_valid.cpu().tolist())
                del outputs, labels_valid, errors, imgs_chunk, labels_chunk, valid_mask, output, pred

            torch.cuda.empty_cache()
        if operation == "mean":
            avg_error_all_labels = []
            for i, label in enumerate(testloader.dataset.labels):
                avg_error = np.mean(all_errors[i])
                avg_error_all_labels.append(avg_error)
                if print_results: print(f"Average Error for {label}: {avg_error:.4f} pixels")
            return np.array(avg_error_all_labels)
        elif operation == "distribution":
            plot_error_distribution(all_errors, fields=testloader.dataset.labels)
            return
        elif operation == "by_field":
            if print_results:
                for i in range(n_fields):
                    arr = torch.tensor(all_errors[i], dtype=torch.float32)
                    agg = getattr(torch, operation)(arr) if operation in dir(torch) else arr.mean()
                    print(f"Head {i}: {operation} error = {agg:.4f}")
            return np.array(all_errors), np.array(all_gts)
        elif operation == "preds":
            return np.array(all_preds), np.array(all_gts)

        

def evaluate_regression_tracker(model, testloader, device, num_steps=10, print_results=True, operation="mean", weighted_avg=False):
        """
        Evaluate the model on the testset.
        
        Parameters:
            testset: A dataset object.
            device: Device to run on ("cuda" or "cpu").
            batch_size: Batch size for evaluation.
            num_steps: Number of simulation time steps to use in the model.
            
        Returns:
            avg_error_x, avg_error_y: Average absolute error for x and y predictions.
        """
        model.eval()  # Set model to evaluation mode
        
        all_errors_x = np.array([])
        all_errors_y = np.array([])

        with torch.no_grad():
            for images, labels in testloader:
                images = images.to(device)
                # Assume labels contain [x, y, ...]; round and convert to int indices
                labels_x = labels[:, 0].to(device)
                labels_y = labels[:, 1].to(device)
                
                # Get model outputs (raw logits for x and y)
                norm_x, norm_y = model(images, num_steps=num_steps)
                if weighted_avg:
                    norm_x, norm_y = logits_to_weighted_avg((norm_x, norm_y))
                height, width = model.input_shape[1], model.input_shape[2]
                preds_x = (norm_x * width).squeeze(1)
                preds_y = (norm_y * height).squeeze(1)
                error_x = torch.abs(preds_x.float() - labels_x.float())
                error_y = torch.abs(preds_y.float() - labels_y.float())
                
                all_errors_x = np.concatenate((all_errors_x, error_x.cpu().numpy()))
                all_errors_y = np.concatenate((all_errors_y, error_y.cpu().numpy()))
        if operation == "mean":
            avg_error_x = np.mean(all_errors_x)
            avg_error_y = np.mean(all_errors_y)
            if print_results: print(f"Average X Error: {avg_error_x:.4f} pixels, Average Y Error: {avg_error_y:.4f} pixels")
            return avg_error_x, avg_error_y
        elif operation == "distribution":
            return all_errors_x, all_errors_y

def evaluate_video_regression_tracker(model, testloader, device, num_steps=10, print_results=True, operation="mean", chunk_size=CHUNK_SIZE, weighted_avg=None):
    model.eval()  # Set model to evaluation mode
    print("Evaluating video regression tracker")
    if weighted_avg is None:
        weighted_avg = getattr(model, "weighted_avg", False)

    n_fields = len(testloader.dataset.labels)
    all_errors = [[] for _ in range(n_fields)]
    max_values = [model.max_values[label] for label in testloader.dataset.labels]
    all_gts = [[] for _ in range(n_fields)]
    all_preds = [[] for _ in range(n_fields)]

    with torch.no_grad():
        for padded_imgs, padded_labels, lengths in testloader:
            # Move everything to device once
            padded_imgs   = padded_imgs.to(device)   # [B, T, C, H, W]
            padded_labels = padded_labels.to(device) # [B, T, …]
            lengths       = lengths.to(device)       # [B]

            # iterate over each chunk of the sequence
            T = padded_imgs.size(1)
            membrane_potentials = []
            for t0 in range(0, T, chunk_size):
                t1 = min(t0 + chunk_size, T)
                imgs_chunk   = padded_imgs[:, t0:t1]   # [B, chunk, C, H, W]
                labels_chunk = padded_labels[:, t0:t1] # [B, chunk, …]
                # adjust lengths for this chunk
                # mask out frames ≥ original length
                frame_idx = torch.arange(t0, t1, device=device).unsqueeze(0)  # [1, chunk]
                valid_mask = frame_idx < lengths.unsqueeze(1)                  # [B, chunk]

                # forward pass on just this chunk
                outputs, membrane_potentials = model((imgs_chunk, lengths), membrane_potentials, num_steps_per_image=num_steps)

                if weighted_avg:
                    outputs = logits_to_weighted_avg(outputs)
                else:
                    outputs = [o.squeeze() for o in outputs]
                # Now outputs is a list of tensors, each of shape [B, chunk_len]
                for i, (output, max_value) in enumerate(zip(outputs, max_values)):
                    labels_valid = labels_chunk[:, :, i][valid_mask]
                    output_valid = output[valid_mask] * max_value
                    errors = torch.abs(output_valid - labels_valid)
                    all_errors[i].extend(errors.cpu().tolist())
                    all_gts[i].extend(labels_valid.cpu().tolist())
                    all_preds[i].extend(output_valid.cpu().tolist())
                del output, labels_valid, output_valid, max_value, errors, imgs_chunk, labels_chunk, outputs, valid_mask
            torch.cuda.empty_cache()
        all_errors = np.array(all_errors)
        if operation == "mean":
            avg_error_all_labels = []
            for i, label in enumerate(testloader.dataset.labels):
                avg_error = np.mean(all_errors[i])
                avg_error_all_labels.append(avg_error)
                if print_results: print(f"Average Error for {label}: {avg_error:.4f} pixels")
            return np.array(avg_error_all_labels)
        elif operation == "distribution":
            plot_error_distribution(all_errors, fields=testloader.dataset.labels)
            return
        elif operation == "by_field":
            if print_results:
                for i in range(n_fields):
                    arr = torch.tensor(all_errors[i], dtype=torch.float32)
                    agg = getattr(torch, operation)(arr) if operation in dir(torch) else arr.mean()
                    print(f"Head {i}: {operation} error = {agg:.4f}")
            return np.array(all_errors), np.array(all_gts)
        elif operation == "preds":
            return np.array(all_preds), np.array(all_gts)


def plot_timestep_curve(model, testloader, device, identifier="", interval = [1, 100, 5]):
    errors = []
    timesteps = []
    for i in range(*interval):
        error = model.evaluate(testloader, device, num_steps=i)
        error_norm = np.linalg.norm(error).item()
        errors.append(error_norm)
        timesteps.append(i)
        print(f"Error at {i} timesteps: {error_norm}")
    # plt.figure()
    plt.plot(timesteps, errors, label=identifier)
    plt.xlabel('Time Steps')
    plt.ylabel('Error')
    plt.title('Error vs Time Steps:')
    # plt.show()


def save_model(model, path = None):
    if path == None: 
        if 'weighted_avg' in model.training_params.keys():
            if model.training_params['weighted_avg']: 
                path = f'models/{model.name}WAvg_q{model.training_params["quantization"]}_{model.training_params["num_steps"]}ts_{model.training_params["num_epochs"]}e.pt'
            else:
                path = f'models/{model.name}_q{model.training_params["quantization"]}_{model.training_params["num_steps"]}ts_{model.training_params["num_epochs"]}e.pt'
        else: path = f'models/{model.name}_q{model.training_params["quantization"]}_{model.training_params["num_steps"]}ts_{model.training_params["num_epochs"]}e.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_params': model.training_params,
    }, path)
    print(f"Model saved at {path}")

def save_best_model(model_type, testloader, device, evaluate = True):
    print("Loading best model...")
    best_model = load_model('models/best_model_current_training.pt', model_type, testloader.dataset, device = device)
    if evaluate:
        print("Best model evaluation...")
        error = best_model.evaluate(testloader, device, num_steps=best_model.training_params['num_steps'], print_results=True)
        error_norm = np.linalg.norm(error).item()
        print(f"Best model error: {error_norm}")
    save_model(best_model)

def measure_inference_time_per_image(model, dataloader, device, num_steps=10, num_batches=10):
    """
    Measure the average inference time per image for a given model and dataloader.
    
    Parameters:
        model: The trained PyTorch model.
        dataloader: DataLoader for the inference data.
        device: torch.device ("cuda" or "cpu").
        num_steps: Number of simulation time steps (if applicable).
        num_batches: Number of batches to average over.
        
    Returns:
        avg_time_per_image: Average inference time (in seconds) per image.
    """
    model.eval()  # Set model to evaluation mode
    per_image_times = []
    
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            images = images.to(device)
            batch_size = images.size(0)
            
            start_time = time.perf_counter()
            _ = model(images, num_steps=num_steps)
            if device == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            # Calculate time per image for this batch
            batch_time = end_time - start_time
            per_image_times.append(batch_time / batch_size)
    
    avg_time_per_image = sum(per_image_times) / len(per_image_times)
    print(f"Average inference time per image over {num_batches} batches: {avg_time_per_image:.6f} seconds")
    return avg_time_per_image

def measure_inference_time_per_image_for_videos_old(model, dataloader, device, num_steps=10, num_batches=10):
    """
    Measure the average inference time per image for a given model and dataloader.
    
    Parameters:
        model: The trained PyTorch model.
        dataloader: DataLoader for the inference data.
        device: torch.device ("cuda" or "cpu").
        num_steps: Number of simulation time steps (if applicable).
        num_batches: Number of batches to average over.
        
    Returns:
        avg_time_per_image: Average inference time (in seconds) per image.
    """
    batch_size = dataloader.batch_size
    if batch_size != 1:
        print("WARNING: Batch size must be 1 for single image inference time calculation.")
    model.eval()  # Set model to evaluation mode
    per_image_times = []
    
    with torch.no_grad():
        for i, (padded_imgs, _, lengths) in enumerate(dataloader):
            if i >= num_batches:
                break
            lengths, sorted_indices = lengths.sort(descending=True)
            lengths = lengths.to(device)
            padded_imgs = padded_imgs[sorted_indices].to(device)


            # print('Padded images shape', padded_imgs.shape)
            # print('Sorted indices', sorted_indices.shape)
            # print('Lengths', lengths.shape)
            images_in_batch = lengths.sum()
            # print('Images in batch', images_in_batch)
            start_time = time.perf_counter()
            _ = model((padded_imgs, lengths), num_steps_per_image=num_steps)
            if device == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            # Calculate time per image for this batch
            batch_time = end_time - start_time
            per_image_times.append(batch_time / images_in_batch)
    
    avg_time_per_image = sum(per_image_times) / len(per_image_times)
    num_batches = min(i+1, num_batches)
    print(f"Average inference time per image over {num_batches} batches: {avg_time_per_image:.6f} seconds")
    return avg_time_per_image

def measure_inference_time_per_image_for_videos(
    model,
    dataloader,
    device,
    num_steps: int = 10,
    num_batches: int = 10,
    chunk_size: int = CHUNK_SIZE
) -> float:
    """
    Measure the average inference time per *valid* image for a video-based model,
    using the same chunked iteration and membrane_potentials state as in your
    evaluate_video_classification_tracker loop.

    Args:
        model:               trained PyTorch model (expects (imgs, lengths), state, num_steps_per_image=...)
        dataloader:          DataLoader yielding (padded_imgs, _, lengths)
        device:              torch.device
        num_steps:           number of simulation steps per image
        num_batches:         how many batches to time
        chunk_size:          max frames to process in one forward-pass chunk

    Returns:
        avg_time_per_image:  average time (s) across all *valid* images
    """
    model.eval()
    per_image_times = []

    with torch.no_grad():
        for batch_idx, (padded_imgs, _, lengths) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            # move to device
            padded_imgs = padded_imgs.to(device)    # [B, T, C, H, W]
            lengths     = lengths.to(device)        # [B]

            # reset state
            membrane_potentials = []

            # timing start
            start_t = time.perf_counter()

            # chunked forward
            B, T, C, H, W = padded_imgs.shape
            for t0 in range(0, T, chunk_size):
                t1 = min(t0 + chunk_size, T)
                imgs_chunk = padded_imgs[:, t0:t1]  # [B, chunk, C, H, W]

                # forward pass
                _, membrane_potentials = model(
                    (imgs_chunk, lengths),
                    membrane_potentials,
                    num_steps_per_image=num_steps
                )

            # ensure all kernels are done
            if device == "cuda":
                torch.cuda.synchronize()

            end_t = time.perf_counter()

            # how many *valid* frames in this batch?
            images_in_batch = lengths.sum().item()
            per_image_times.append((end_t - start_t) / images_in_batch)
            del imgs_chunk, membrane_potentials, lengths, padded_imgs

    avg_time = sum(per_image_times) / len(per_image_times)
    print(f"Average inference time per image over {len(per_image_times)} batches: "
          f"{avg_time:.6f} s")
    return avg_time

def load_model(path, model_class, trainset, device):
    checkpoint = torch.load(path)
    # Create model instance with the same parameters as the saved model
    if checkpoint['training_params'].keys().__contains__('weighted_avg'):
        print(f"Loading model with weighted average: {checkpoint['training_params']['weighted_avg']}")
        model = model_class(trainset, weighted_avg=checkpoint['training_params']['weighted_avg'])
    else:
        model = model_class(trainset, weighted_avg=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.training_params = checkpoint['training_params']
    model.to(device)
    return model

def get_preds_video_regression(model, video, length, labels, device, num_steps=10, weighted_avg = False): # This one shows also the prediction from the model
    """Generator that yields images, labels, and predictions given one sequence of images."""
    model.eval()
    membrane = []
    with torch.no_grad():
        outputs, membrane = model((video.unsqueeze(0).to(device), torch.tensor([length])), membrane, num_steps_per_image=num_steps)
        if weighted_avg:
            outputs = logits_to_weighted_avg(outputs)
        preds = []
        max_values = [model.max_values[label] for label in labels]
        for _, (output, max_val) in enumerate(zip(outputs, max_values)):
            pred = output * max_val
            preds.append(pred.squeeze(0).cpu().numpy())
        preds = np.array(preds)
        print(preds.shape)
    return preds

def get_preds_all_videos(
    model,
    dataloader,
    device,
    num_steps: int = 10,
    weighted_avg: bool = False,
    chunk_size: int = CHUNK_SIZE
):
    """
    Returns:
      all_preds  : list of length N, each an array of shape [T_i, H]
      all_labels : list of length N, each an array of shape [T_i, H]

    where N = number of videos, T_i = true length of video i,
    H = number of heads (e.g. x, y, in_fov, ...).
    """
    model.eval()
    all_preds  = []
    # how many heads?
    H = len(dataloader.dataset.labels)

    with torch.no_grad():
        for padded_imgs, padded_labels, lengths in tqdm(dataloader, desc="Predicting"):
            B, T = padded_imgs.shape[:2]

            imgs   = padded_imgs.to(device)          # [B, T, C, H, W]
            lengths = lengths.to(device)             # [B]

            # container for this batch: [B, H, T]
            preds_batch = torch.zeros((B, H, T), dtype=torch.long, device=device)
            mems = []  # reset membrane potentials at start of batch

            # forward in chunks
            for t0 in range(0, T, chunk_size):
                t1 = min(t0 + chunk_size, T)
                chunk = imgs[:, t0:t1]               # [B, chunk, C, H, W]

                logits_list, mems = model(
                    (chunk, lengths),
                    mems,
                    num_steps_per_image=num_steps
                )
                # logits_list is a list of length H:
                #   logits_list[h] has shape [B, n_bins_h, chunk]
                for h, logits in enumerate(logits_list):
                    # class prediction per time-step
                    if weighted_avg:
                        pred_chunk = logits_to_weighted_avg(logits)
                    else:
                        pred_chunk = torch.argmax(logits, dim=1)   # [B, chunk]
                    preds_batch[:, h, t0:t1] = pred_chunk

            # now split out each video up to its true length
            # create a boolean mask [B, T]
            mask_full = (torch.arange(T, device=device)[None, :] 
                         < lengths[:, None])

            for b in range(B):
                L = lengths[b].item()
                mask = mask_full[b]  # [T] boolean

                # take only the valid time-steps
                preds_i = preds_batch[b, :, mask].cpu().numpy()   # [H, L]

                all_preds.append(preds_i)

    return all_preds # (N, H, T)




def get_preds_video_classification(model, video, length, labels, device, num_steps=20): # This one shows also the prediction from the model
    model.eval()
    with torch.no_grad():
        outputs, _ = model((video.unsqueeze(0).to(device), torch.tensor([length])), [], num_steps_per_image=num_steps)
        preds = []
        for _, output in enumerate(outputs):
            print(output.shape)
            pred = torch.argmax(output, dim=1)
            print(pred.shape)
            preds.extend(pred.cpu().numpy())
        preds = np.array(preds)
        print(preds.shape)
    return preds

def invert_argmax(indices: torch.LongTensor,
                  num_classes: int = None,
                  dim: int = -1,
                  dtype=torch.float32) -> torch.Tensor:
    """
    Given an integer tensor of indices, return a tensor of scores such that
    `scores.argmax(dim)` == `indices`.

    Args:
        indices    (LongTensor): input tensor of indices.
        num_classes (int, optional): number of classes. If None, it will be inferred
                                     as indices.max()+1.
        dim         (int):       dimension along which to one‐hot / argmax.
        dtype       (torch.dtype): dtype of the output scores.

    Returns:
        Tensor of shape indices.shape[:dim] + [num_classes] + indices.shape[dim+1:]
        where out[i,…,j,…] = 1 if j == indices[i,…] else 0.
    """
    if num_classes is None:
        num_classes = int(indices.max().item()) + 1

    # Build output shape
    out_shape = list(indices.shape)
    out_shape.insert(dim if dim >= 0 else dim + indices.dim() + 1, num_classes)

    # Create empty and scatter 1s at target locations
    out = torch.zeros(*out_shape, dtype=dtype, device=indices.device)
    # expand indices to have a size-1 axis at `dim`
    idx_expanded = indices.unsqueeze(dim)
    out.scatter_(dim, idx_expanded, 1.0)
    return out

def freeze_all_except_given(model, layer_name):
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(f"fc_layers.{layer_name}") or name.startswith(f"lif_layers.{layer_name}")
    print("Trainable parameters:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print("✓", name)

def freeze_given(model, layer_name):
    for name, param in model.named_parameters():
        if name.startswith(f"fc_layers.{layer_name}") or name.startswith(f"lif_layers.{layer_name}"):
            param.requires_grad = False
        else:
            param.requires_grad = True
    print("Trainable parameters:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print("✓", name)

def unfreeze_all(model):
    for name, param in model.named_parameters():
        param.requires_grad = True
    print("All parameters are trainable.")