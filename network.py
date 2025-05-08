
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

MAX_VAL_R_CAM = dt.MAX_VAL_R_CAM
MAX_VAL_Y_CAM = dt.MAX_VAL_Y_CAM
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


class SCNN_Tracker2L(nn.Module):
    def __init__(self, image_shape, bins_factor, beta=0.8, learn_threshold = False):
        """
        input_shape: tuple (channels, height, width) of input event data.
        """
        super(SCNN_Tracker2L, self).__init__()
        self.model_type = 'Tracker2L'
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
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2)   # Expected: from (2, H, W) to (16, H/2, W/2)
        self.lif1 = snn.Leaky(beta)
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # Expected: (32, H/4, W/4)
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
        probs_x = F.softmax(outputs_x, dim=1)
        probs_y = F.softmax(outputs_y, dim=1)
        
        return probs_x, probs_y#, [spk_x_rec, spk_y_rec, mem_x_rec, mem_y_rec]
    
    def start_training(self, trainloader, optimizer, device, validationloader = None, num_steps = 10, num_epochs=20, plot = True):
        batch_size = trainloader.batch_size
        loss_function = classification_loss
        self.training_params = {
            "type": self.model_type,
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

class SCNN_Tracker3L(nn.Module):
    def __init__(self, image_shape, bins_factor, beta=0.8, learn_threshold = False):
        """
        input_shape: tuple (channels, height, width) of input event data.
        """
        super(SCNN_Tracker3L, self).__init__()

        self.model_type = 'Tracker3L'
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
            "type": self.model_type,
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

class SCNN_Tracker3LWAvg(nn.Module):
    def __init__(self, image_shape, bins_factor, beta=0.8, learn_threshold = False):
        """
        input_shape: tuple (channels, height, width) of input event data.
        """
        super(SCNN_Tracker3LWAvg, self).__init__()

        self.model_type = 'Tracker3LWAvg'
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
            "type": self.model_type,
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

class SCNN_TrackerRegression(nn.Module):
    def __init__(self, image_shape, beta=0.8, learn_threshold = False):
        """
        input_shape: tuple (channels, height, width) of input event data.
        """
        super(SCNN_TrackerRegression, self).__init__()

        self.model_type = 'TrackerRegression'
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
        
        # Apply sigmoid to obtain normalized coordinates between 0 and 1.
        norm_x = torch.sigmoid(outputs_x)
        norm_y = torch.sigmoid(outputs_y)
            
        return norm_x, norm_y#, [spk_x_rec, spk_y_rec, mem_x_rec, mem_y_rec]
    
    def start_training(self, trainloader, optimizer, device, validationloader = None, num_steps = 10, num_epochs=20, plot = True):
        loss_function = regression_loss
        self.training_params = {
            "type": self.model_type,
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

        self.model_type = 'TrackerClassGASP'
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
            "type": self.model_type,
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
    
class SCNN_Video_Tracker_Class_OLD(nn.Module): # Based on the SCNN_Tracker3L
    def __init__(self, trainset, beta=0.8, learn_threshold = False, weighted_avg = False):
        """
        input_shape: tuple (channels, height, width) of input event data.
        """
        super(SCNN_Video_Tracker_Class, self).__init__()

        self.model_type = 'VideoTrackerClass'
        self.beta = beta
        self.learn_threshold = learn_threshold
        self.image_shape = trainset.image_shape
        self.bins_factor = trainset.quantization / trainset.label_quantization
        self.weighted_avg = weighted_avg
        self.g = 9.81  # Gravitational constant for the simulation
        self.dt = 0.01  # Time step for the simulation
        self.max_values = {
            'x_cam': self.image_shape[2],
            'y_cam': self.image_shape[1],
            'R_cam': MAX_VAL_R_CAM
        }

        # Convolutional layers (assuming 2-channel input for polarity split)
        channels, y_pixels, x_pixels = self.image_shape
        x_bins, y_bins = int(x_pixels*self.bins_factor), int(y_pixels*self.bins_factor)

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
        self.fc_x = nn.Linear(self.flattened_size, x_bins)
        self.lif_x = snn.Leaky(beta, learn_threshold=learn_threshold)
        self.fc_y = nn.Linear(self.flattened_size, y_bins)
        self.lif_y = snn.Leaky(beta, learn_threshold=learn_threshold)
        print(f"Number of x and y bins: {x_bins}, {y_bins}")
        for i, label in enumerate(trainset.labels):
            self.fc_[i] = nn.Linear(self.flattened_size, self.max_values[label])
            


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
        mem_x = self.lif_x.init_leaky()
        mem_y = self.lif_y.init_leaky()

        # Record final layer
        # Initialize tensors to store outputs for each time step per sequence.
        outputs_seq_x = torch.zeros(batch_size, self.fc_x.out_features, max_seq_len, device=padded_x.device)
        outputs_seq_y = torch.zeros(batch_size, self.fc_y.out_features, max_seq_len, device=padded_x.device)

        for i in range(max_seq_len):
            # Create a boolean mask for samples that have a valid image at time t
            valid_mask = (i < lengths).to(padded_x.device)
            if valid_mask.sum() == 0:
                break  # No valid data at this time step for any sequence

            # Get the images for the valid sequences at time t (shape: [valid_batch, channels, height, width])
            x_t = padded_x[valid_mask, i]

            outputs_x = 0
            outputs_y = 0
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
                out_x = self.fc_x(s3_flat)
                spk_x, mem_x = self.lif_x(out_x, mem_x)
                out_y = self.fc_y(s3_flat)
                spk_y, mem_y = self.lif_y(out_y, mem_y)

                outputs_x += out_x
                outputs_y += out_y

                # Delete intermediate tensors
                del x1, spk1, x2, spk2, x3, spk3, s3_flat, out_x, out_y, spk_x, spk_y
            if padded_x.device == "cuda": torch.cuda.empty_cache()
            # Average over time steps
            outputs_x = outputs_x / num_steps_per_image
            outputs_y = outputs_y / num_steps_per_image

            # print('Outputs x and valid mask', outputs_x.shape, valid_mask.shape)

            outputs_seq_x[valid_mask, :, i] = outputs_x
            outputs_seq_y[valid_mask, :, i] = outputs_y

        
        # Apply softmax to get probabilities
        # probs_x = F.softmax(outputs_x, dim=1)
        # probs_y = F.softmax(outputs_y, dim=1)
        
        return outputs_seq_x, outputs_seq_y#, [spk_x_rec, spk_y_rec, mem_x_rec, mem_y_rec]
    
    def start_training(self, trainloader, optimizer, device, validationloader = None, num_steps = 10, num_epochs=20, plot = True):
        batch_size = trainloader.batch_size
        if self.weighted_avg:
            loss_function = regression_loss
        else: 
            loss_function = classification_loss

        self.training_params = {
            "type": self.model_type,
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
        
        training_loop_videos(self, trainloader, optimizer, device, loss_function, validationloader, num_steps, num_epochs, plot=plot)
    
    def evaluate(self, testloader, device, num_steps, print_results=False, operation = 'mean'):
        return evaluate_video_regression_tracker(self, testloader, device, num_steps, print_results, operation) if self.weighted_avg else evaluate_video_classification_tracker(self, testloader, device, num_steps, print_results, operation)

class SCNN_Video_Tracker_Class(nn.Module): # Based on the SCNN_Tracker3L
    def __init__(self, trainset, beta=0.8, learn_threshold = False, weighted_avg = False):
        """
        input_shape: tuple (channels, height, width) of input event data.
        """
        super(SCNN_Video_Tracker_Class, self).__init__()

        self.model_type = 'VideoTrackerClass'
        self.beta = beta
        self.learn_threshold = learn_threshold
        self.image_shape = trainset.image_shape
        self.bins_factor = trainset.quantization / trainset.label_quantization
        self.weighted_avg = weighted_avg
        self.g = 9.81  # Gravitational constant for the simulation
        self.dt = 0.01  # Time step for the simulation
        self.max_values = {
            'x_cam': self.image_shape[2],
            'y_cam': MAX_VAL_Y_CAM,
            'R_cam': MAX_VAL_R_CAM
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

        # Dynamically compute the flattened feature size by passing a dummy input.
        self.flattened_size = self._get_flattened_size(input_shape)
        print(f"Flattened feature size: {self.flattened_size}")

        self.fc_layers = nn.ModuleDict()
        self.lif_layers = nn.ModuleDict()
        
        for i, label in enumerate(trainset.labels):
            n_bins = int(self.max_values[label] * self.bins_factor)
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
        outputs_seq = []
        for i in range(len(self.labels)):
            outputs_seq.append(torch.zeros(batch_size, self.n_bins[i], max_seq_len, device=padded_x.device))

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
                
                for j, label in enumerate(self.labels):
                    fc_out = self.fc_layers[label](s3_flat)
                    # spk_out, _ = self.lif_layers[label](fc_out)
                    outputs[j] += fc_out

                # Delete intermediate tensors
                del x1, spk1, x2, spk2, x3, spk3, s3_flat
            if padded_x.device == "cuda": torch.cuda.empty_cache()
            # Average over time steps
            for j in range(len(self.labels)):
                outputs[j] = outputs[j] / num_steps_per_image
                outputs_seq[j][valid_mask, :, i] = outputs[j]

            # print('Outputs x and valid mask', outputs_x.shape, valid_mask.shape)

        
        # Apply softmax to get probabilities
        # probs_x = F.softmax(outputs_x, dim=1)
        # probs_y = F.softmax(outputs_y, dim=1)
        
        return outputs_seq#, [spk_x_rec, spk_y_rec, mem_x_rec, mem_y_rec]
    
    def start_training(self, trainloader, optimizer, device, validationloader = None, num_steps = 10, num_epochs=20, plot = True):
        batch_size = trainloader.batch_size
        if self.weighted_avg:
            loss_function = regression_loss
        else: 
            loss_function = classification_loss

        self.training_params = {
            "type": self.model_type,
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
        
        training_loop_videos(self, trainloader, optimizer, device, loss_function, validationloader, num_steps, num_epochs, plot=plot)
    
    def evaluate(self, testloader, device, num_steps, print_results=False, operation = 'mean'):
        return evaluate_video_regression_tracker(self, testloader, device, num_steps, print_results, operation) if self.weighted_avg else evaluate_video_classification_tracker(self, testloader, device, num_steps, print_results, operation)


class PISNN(nn.Module): # Based on the SCNN_Tracker3L
    def __init__(self, trainset, beta=0.8, learn_threshold = False, weighted_avg = False):
        """
        input_shape: tuple (channels, height, width) of input event data.
        """
        super(PISNN, self).__init__()

        self.model_type = 'PISNN'
        self.beta = beta
        self.learn_threshold = learn_threshold
        self.image_shape = trainset.image_shape
        self.bins_factor = trainset.quantization / trainset.label_quantization
        self.weighted_avg = weighted_avg
        self.g = 9.81  # Gravitational acceleration in m/s^2
        self.dt = 0.01  # Time step in seconds (adjust as needed)
        self.max_values = {
            'x_cam': self.image_shape[2],
            'y_cam': MAX_VAL_Y_CAM,
            'R_cam': MAX_VAL_R_CAM
        }

        # Convolutional layers (assuming 2-channel input for polarity split)
        channels, y_pixels, x_pixels = self.image_shape
        x_bins, y_bins = int(x_pixels*self.bins_factor), int(y_pixels*self.bins_factor)

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
                out_x = self.fc_x(s3_flat)
                out_y = self.fc_y(s3_flat)
                out_z = self.fc_z(s3_flat)

                outputs_x += out_x
                outputs_y += out_y
                outputs_z += out_z

                # Delete intermediate tensors
                del x1, spk1, x2, spk2, x3, spk3, s3_flat, out_x, out_y
            if padded_x.device == "cuda": torch.cuda.empty_cache()
            # Average over time steps
            outputs_x = outputs_x / num_steps_per_image
            outputs_y = outputs_y / num_steps_per_image
            outputs_z = outputs_z / num_steps_per_image

            norm_x = torch.sigmoid(outputs_x)
            norm_y = torch.sigmoid(outputs_y)
            norm_z = torch.sigmoid(outputs_z)

            # print('Outputs x and valid mask', outputs_x.shape, valid_mask.shape)

            outputs_seq_x[valid_mask, :, i] = norm_x
            outputs_seq_y[valid_mask, :, i] = norm_y
            outputs_seq_z[valid_mask, :, i] = norm_z

        
        return outputs_seq_x, outputs_seq_y, outputs_seq_z
    
    def start_training(self, trainloader, optimizer, device, loss_function, validationloader = None, num_steps = 10, num_epochs=20, plot = True):
        batch_size = trainloader.batch_size

        self.training_params = {
            "type": self.model_type,
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
        
        training_loop_videos(self, trainloader, optimizer, device, loss_function, validationloader, num_steps, num_epochs, plot=plot)
    
    def evaluate(self, testloader, device, num_steps, print_results=False, operation = 'mean'):
        return evaluate_video_regression_tracker(self, testloader, device, num_steps, print_results, operation)


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


def training_loop_videos_OLD(model, trainloader, optimizer, device, loss_function, validationloader = None, num_steps = 10, num_epochs=20, plot = True):
    print('Training loop videos')
    print('Model:', model.model_type)
    print('Batch size:', trainloader.batch_size)
    label_fields = trainloader.dataset.labels
    print('Labels:', labels)
    print('Quantization:', trainloader.dataset.quantization)
    epoch_losses = []
    validation_errors = []
    max_values = [model.max_values[field] for field in label_fields]
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for padded_imgs, padded_labels, lengths in trainloader:
            # Sort the batch by sequence lengths in descending order
            lengths, sorted_indices = lengths.sort(descending=True)
            lengths = lengths.to(device)
            padded_imgs = padded_imgs[sorted_indices].to(device)
            labels = padded_labels[sorted_indices].to(device)
            
            optimizer.zero_grad()
            logits = model((padded_imgs, lengths), num_steps_per_image=num_steps)
            
            loss = loss_function(model, logits, labels, lengths = lengths, max_values = max_values)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            del padded_imgs, padded_labels, logits, loss
            if device == "cuda": torch.cuda.empty_cache()
        
        avg_loss = epoch_loss / len(trainloader)
        epoch_losses.append(avg_loss)
        if validationloader is not None:
            error = model.evaluate(validationloader, device, num_steps, print_results=False)
            error = np.linalg.norm(error).item()
            validation_errors.append(error)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Validation Error: {error:.4f} pixels")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")       
    
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

def training_loop_videos(model, trainloader, optimizer, device, loss_function, validationloader = None, num_steps = 10, num_epochs=20, plot = True,
                         chunk_size=CHUNK_SIZE):
    epoch_losses = []
    validation_errors = []
    max_values = [model.max_values[label] for label in trainloader.dataset.labels]
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
            for t0 in range(0, T, chunk_size):
                t1 = min(t0 + chunk_size, T)
                imgs_chunk   = padded_imgs[:, t0:t1]   # [B, chunk, C, H, W]
                labels_chunk = padded_labels[:, t0:t1] # [B, chunk, …]
                # adjust lengths for this chunk
                # mask out frames ≥ original length
                frame_idx = torch.arange(t0, t1, device=device).unsqueeze(0)  # [1, chunk]
                valid_mask = frame_idx < lengths.unsqueeze(1)                  # [B, chunk]

                # forward pass on just this chunk
                logits = model((imgs_chunk, lengths), num_steps_per_image=num_steps)
                loss  = loss_function(model, logits, labels_chunk, mask=valid_mask, max_values=max_values)


                # Accumulate grads over chunks and step once after the inner loop:
                loss.backward()
                epoch_loss += loss.item()
                total_chunks += 1

                # free up memory
                del imgs_chunk, labels_chunk, logits, loss, valid_mask

            # Step here:
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
        avg_loss = epoch_loss / trainloader.dataset.return_n_frames()
        epoch_losses.append(avg_loss)
        model.training_params["num_epochs"] += 1

        if validationloader is not None:
            error = model.evaluate(validationloader, device, num_steps, print_results=False)
            error = np.linalg.norm(error).item()
            validation_errors.append(error)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Validation Error: {error:.4f} pixels")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

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


########################################################################
#                       LOSS FUNCTIONS                                 #
########################################################################        


def classification_loss_OLD(_, logits_x, logits_y, labels_x, labels_y, lengths = None):
    criterion = nn.CrossEntropyLoss()
    if lengths == None:
        loss_x = criterion(logits_x, labels_x.round().long())
        loss_y = criterion(logits_y, labels_y.round().long())
    else:
        batch, num_classes_x, max_seq_length = logits_x.shape
        # Create a mask: for each sequence in the batch, valid if t < length
        # The mask will have shape [batch, max_seq_length]
        mask = torch.arange(max_seq_length, device=logits_x.device).expand(batch, max_seq_length) < lengths.unsqueeze(1)              # [batch * max_seq_length]
        labels_x_valid = labels_x[mask]
        labels_y_valid = labels_y[mask]
        logits_x = logits_x.transpose(2, 1)
        logits_y = logits_y.transpose(2, 1)
        logits_x_valid = logits_x[mask, :]
        logits_y_valid = logits_y[mask, :]
        loss_x = criterion(logits_x_valid, labels_x_valid.round().long())
        loss_y = criterion(logits_y_valid, labels_y_valid.round().long())
    return loss_x + loss_y

def classification_loss(model, logits, labels, max_values = None, mask = None): # Max_values is not used here, but it is used in the regression loss
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    if mask == None:
        # loss_x = criterion(logits_x, labels_x.round().long())
        # loss_y = criterion(logits_y, labels_y.round().long())
        for i, logit in enumerate(logits):
            loss = criterion(logit, labels[:,i].round().long())
            total_loss += loss
        del loss, logit
    else:
        # batch, num_classes_x, max_seq_length = logits[0].shape
        # Create a mask: for each sequence in the batch, valid if t < length
        # The mask will have shape [batch, max_seq_length]
        # mask = torch.arange(max_seq_length, device=logits[0].device).expand(batch, max_seq_length) < lengths.unsqueeze(1)              # [batch * max_seq_length]
        for i, logit in enumerate(logits):
            labels_valid = labels[mask][:, i]
            logit = logit.transpose(2, 1)
            logit_valid = logit[mask]
            loss = criterion(logit_valid, labels_valid.round().long())
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

def weighted_avg_loss_OLD(model, logits_x, logits_y, labels_x, labels_y, lengths = None):
    criterion = nn.MSELoss()
    if lengths == None: # This doesnt work for images
        weighted_x, weighted_y = logits_to_weighted_avg(logits_x, logits_y)
        height, width = model.image_shape[1], model.image_shape[2]
        weighted_x = weighted_x * width
        weighted_y = weighted_y * height
        loss_x = criterion(weighted_x, labels_x.unsqueeze(1))
        loss_y = criterion(weighted_y, labels_y.unsqueeze(1))
    else:
        weighted_x, weighted_y = logits_to_weighted_avg(logits_x, logits_y)

        batch, num_classes_x, max_seq_length = logits_x.shape
        height, width = model.image_shape[1], model.image_shape[2]

        # Create a mask: for each sequence in the batch, valid if t < length
        # The mask will have shape [batch, max_seq_length]
        mask = torch.arange(max_seq_length, device=weighted_x.device).expand(batch, max_seq_length) < lengths.unsqueeze(1)              # [batch * max_seq_length]
        labels_x_valid = labels_x[mask]
        labels_y_valid = labels_y[mask]
        weighted_x_valid = weighted_x[mask] * width
        weighted_y_valid = weighted_y[mask] * height
        loss_x = criterion(weighted_x_valid, labels_x_valid) 
        loss_y = criterion(weighted_y_valid, labels_y_valid)
    return loss_x + loss_y

def pinn_loss_OLD(model,
              preds_x, preds_y, preds_z,
              labels_x, labels_y, labels_z,
              lengths=None):
    """
    Physics‐Informed 3D loss with soft flight↔roll transition.

    Args:
      model:      PINN with .dt (time step), .g (gravity), .mu (friction coeff)
      preds_*:    (B, T) predicted x, y, z positions
      labels_*:   (B, T) ground‐truth x, y, z positions
      lengths:    (B,) valid sequence lengths

    Returns:
      L_data + L_phys
    """
    if lengths is None:
        raise ValueError("Provide 'lengths' for variable‐length video sequences.")

    B, T = preds_x.shape
    device = preds_x.device

    # 1) Data loss (mask out padded frames)
    mask = (torch.arange(T, device=device)
            .unsqueeze(0).expand(B, T)
            < lengths.unsqueeze(1))  # (B, T)
    x_p = preds_x[mask]
    y_p = preds_y[mask]
    z_p = preds_z[mask]
    x_t = labels_x[mask]
    y_t = labels_y[mask]
    z_t = labels_z[mask]
    L_data = torch.mean((x_p - x_t)**2 + (y_p - y_t)**2 + (z_p - z_t)**2)

    # 2) Physics residuals via finite differences
    pos = torch.stack([preds_x, preds_y, preds_z], dim=1)  # (B, 3, T)
    dt = model.dt

    # second-order diff for acceleration: (B,3,T-2)
    dd = (pos[:, :, 2:] - 2*pos[:, :, 1:-1] + pos[:, :, :-2]) / (dt**2)
    # first-order diff for velocity: (B,3,T-2)
    dv = (pos[:, :, 2:] - pos[:, :, :-2]) / (2*dt)

    # mask valid physics frames where t+2 < length
    phys_mask = (torch.arange(T-2, device=device)
                 .unsqueeze(0).expand(B, T-2)
                 < (lengths - 2).unsqueeze(1))  # (B,T-2)

    # flatten masked residual entries
    dd_x = dd[:, 0, :][phys_mask]
    dd_y = dd[:, 1, :][phys_mask]
    dd_z = dd[:, 2, :][phys_mask]
    dv_x = dv[:, 0, :][phys_mask]
    dv_y = dv[:, 1, :][phys_mask]
    dv_z = dv[:, 2, :][phys_mask]

    # 3) Soft gate between flight & roll based on height z
    z_mid = pos[:, 2, 1:-1]                       # (B,T-2)
    eps, k = 0.01, 100.0
    s_f = torch.sigmoid(k * (z_mid - eps))[phys_mask]  # flight weight
    s_r = 1.0 - s_f                                    # roll weight

    # 4) Flight residual: ddx≈0, ddy≈0, ddz+g≈0
    r_f = dd_x**2 + dd_y**2 + (dd_z + model.g)**2

    # 5) Roll residual: contact z≈0, vz≈0 + friction on x,y
    contact = (z_mid[phys_mask])**2 + dv_z**2
    friction_x = (dv_x + model.mu * model.g * torch.sign(dv_x))**2
    friction_y = (dv_y + model.mu * model.g * torch.sign(dv_y))**2
    r_r = contact + friction_x + friction_y

    L_phys = torch.mean(s_f * r_f + s_r * r_r)

    return L_data + L_phys

def pinn_loss(model,
              outputs,             # what your model returns for this chunk
              labels,             # labels_chunk: shape [B, chunk_len, 3]
              mask,               # valid_mask: shape [B, chunk_len]
              max_values=None):   # passed in but not needed here

    """
    Physics‐Informed 3D loss for one chunk, given:
      - model: PINN with .dt, .g, .mu
      - logits: tuple (preds_x, preds_y, preds_z), each [B, chunk_len]
      - labels: tensor [B, chunk_len, 3]  (columns: x,y,z)
      - mask:   boolean tensor [B, chunk_len] marking real frames
      - max_values: ignored here (only needed for other losses)
    """
    preds = [(o * m).squeeze() for o, m in zip(outputs, max_values)]
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
    s_f = torch.sigmoid(k * (z_mid - BALL_RADIUS - eps))
    s_r = 1 - torch.sigmoid(k * (z_mid - BALL_RADIUS - 0.01))

    # flight residual
    r_f = dd_x**2 + dd_y**2 + (dd_z + model.g)**2

    # roll residual
    contact    = (z_mid-BALL_RADIUS)**2 + dv_z**2
    # friction_x = (dv_x + model.mu * model.g * torch.sign(dv_x))**2
    # friction_y = (dv_y + model.mu * model.g * torch.sign(dv_y))**2
    r_r        = contact #+ friction_x + friction_y

    L_phys = torch.mean(s_f * r_f + s_r * r_r)

    return L_data + L_phys


def logits_to_weighted_avg_OLD(logits_x, logits_y):

    probs_x = torch.softmax(logits_x, dim=1)  
    probs_y = torch.softmax(logits_y, dim=1)
    batch_size = probs_x.shape[0]

    # Compute the weighted sum (expected value):
    # Create position tensors for x and y
    positions_x = torch.linspace(0, 1, steps = probs_x.shape[1], device=probs_x.device).unsqueeze(0)
    positions_y = torch.linspace(0, 1, steps = probs_y.shape[1], device=probs_y.device).unsqueeze(0)

    # Adjust for batch size
    positions_x = positions_x.expand(batch_size, -1).unsqueeze(2)  # shape: [batch_size, x_bins, 1]
    positions_y = positions_y.expand(batch_size, -1).unsqueeze(2)  # shape: [batch_size, y_bins, 1]

    # Multiply each prob dist with corresponding position and sum
    outputs_seq_x = (probs_x * positions_x).sum(dim=1)  # shape: [batch_size, time]
    outputs_seq_y =  (probs_y * positions_y).sum(dim=1)  # shape: [batch_size, time]
    return outputs_seq_x, outputs_seq_y

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

def plot_error_distribution(errors_x, errors_y, n_bins=5):
    
    errors = np.array([errors_x, errors_y])
    errors = np.linalg.norm(errors, axis=0)
 
    # Add epsilon to avoid zero values
    epsilon = 1e-9
    errors_safe = errors + epsilon

    # Fit normal distributions to X/Y errors
    mu_x, std_x = norm.fit(errors_x)
    mu_y, std_y = norm.fit(errors_y)

    # Fit Rayleigh distribution to total error
    rayleigh_params = rayleigh.fit(errors_safe, floc=0)  # Fix location at 0
    scale_total = rayleigh_params[1]  # Scale parameter (σ) of Rayleigh distribution

    # Calculate 95% confidence intervals
    confidence_level = 0.95
    z_score = norm.ppf(confidence_level)  # For normal distributions
    ci_x = mu_x + z_score * std_x
    ci_y = mu_y + z_score * std_y
    ci_total = rayleigh.ppf(confidence_level, scale=scale_total)  # Rayleigh CI

    print(f"X errors: μ={mu_x:.2f}, σ={std_x:.2f}, 95% CI={ci_x:.3f} pixels")
    print(f"Y errors: μ={mu_y:.2f}, σ={std_y:.2f}, 95% CI={ci_y:.3f} pixels")
    print(f"Total errors: σ={scale_total:.2f}, 95% CI={ci_total:.3f} pixels")

    # Create plot
    plt.figure(figsize=(12, 6))

    # Plot histograms
    plt.hist(errors_x, bins=n_bins, alpha=0.5, density=True, label='X errors', color='red')
    plt.hist(errors_y, bins=n_bins, alpha=0.5, density=True, label='Y errors', color='green')
    plt.hist(errors, bins=n_bins, alpha=0.5, density=True, label='Total errors', color='blue')

    # Plot PDFs
    x_x = np.linspace(min(errors_x), max(errors_x), 1000)
    x_y = np.linspace(min(errors_y), max(errors_y), 1000)
    x_total = np.linspace(0, max(errors), 1000)  # Rayleigh starts at 0

    plt.plot(x_x, norm.pdf(x_x, mu_x, std_x), 'r--', lw=2, 
            label=f'X fit: μ={mu_x:.2f}, σ={std_x:.2f}')
    plt.plot(x_y, norm.pdf(x_y, mu_y, std_y), 'g--', lw=2, 
            label=f'Y fit: μ={mu_y:.2f}, σ={std_y:.2f}')
    plt.plot(x_total, rayleigh.pdf(x_total, scale=scale_total), 'b--', lw=2, 
            label=f'Total fit: σ={scale_total:.2f}')

    # Plot 95% CI lines
    plt.axvline(ci_x, color='red', linestyle=':', linewidth=2, 
                label=f'X 95% CI: {ci_x:.3f}')
    plt.axvline(ci_y, color='green', linestyle=':', linewidth=2, 
                label=f'Y 95% CI: {ci_y:.3f}')
    plt.axvline(ci_total, color='blue', linestyle=':', linewidth=2, 
                label=f'Total 95% CI: {ci_total:.3f}')

    plt.xlabel('Error')
    plt.ylabel('Probability Density')
    plt.title('Error Distribution with 95% Confidence Intervals')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()


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
            plot_error_distribution(all_errors_x, all_errors_y)
            return all_errors_x, all_errors_y

def evaluate_video_classification_tracker(model, testloader, device, num_steps=10, print_results=True, operation="mean"):
    model.eval()  # Set model to evaluation mode

    all_errors_x = np.array([])
    all_errors_y = np.array([])

    with torch.no_grad():
        for padded_imgs, padded_labels, lengths in testloader:
            # Sort the batch by sequence lengths in descending order
            lengths, sorted_indices = lengths.sort(descending=True)
            lengths = lengths.to(device)
            padded_imgs = padded_imgs[sorted_indices].to(device)
            padded_labels = padded_labels[sorted_indices]
            # Use only x and y coordinates for classification.
            labels_x = padded_labels[:, :, 0].round().long().to(device)
            labels_y = padded_labels[:, :, 1].round().long().to(device)
            
            probs_x, probs_y = model((padded_imgs, lengths), num_steps_per_image=num_steps)
            
            batch, num_classes_x, max_seq_length = probs_x.shape

            # Create a mask: for each sequence in the batch, valid if t < length
            # The mask will have shape [batch, max_seq_length]
            mask = torch.arange(max_seq_length, device=probs_x.device).expand(batch, max_seq_length) < lengths.unsqueeze(1)              # [batch * max_seq_length]
            labels_x_valid = labels_x[mask]
            labels_y_valid = labels_y[mask]
            probs_x = probs_x.transpose(2, 1)
            probs_y = probs_y.transpose(2, 1)
            probs_x_valid = probs_x[mask, :]
            probs_y_valid = probs_y[mask, :]
            # Use argmax to get the predicted bin (x and y coordinates)
            preds_x = torch.argmax(probs_x_valid, dim=1)
            preds_y = torch.argmax(probs_y_valid, dim=1)
            
            # Compute absolute error per sample
            error_x = torch.abs(preds_x.float() - labels_x_valid.float())
            error_y = torch.abs(preds_y.float() - labels_y_valid.float())

            all_errors_x = np.concatenate((all_errors_x, error_x.cpu().numpy()))
            all_errors_y = np.concatenate((all_errors_y, error_y.cpu().numpy()))
        if operation == "mean":
            avg_error_x = np.mean(all_errors_x)
            avg_error_y = np.mean(all_errors_y)
            if print_results: print(f"Average X Error: {avg_error_x:.4f} pixels, Average Y Error: {avg_error_y:.4f} pixels")
            return avg_error_x, avg_error_y
        elif operation == "distribution":
            plot_error_distribution(all_errors_x, all_errors_y)
            return all_errors_x, all_errors_y
        

def evaluate_regression_tracker(model, testloader, device, num_steps=10, print_results=True, operation="mean"):
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

def evaluate_video_regression_tracker(model, testloader, device, num_steps=10, print_results=True, operation="mean", chunk_size=40):
    model.eval()  # Set model to evaluation mode

    all_errors = []
    max_values = [model.max_values[label] for label in testloader.dataset.labels]

    with torch.no_grad():
        for padded_imgs, padded_labels, lengths in testloader:
            # Move everything to device once
            padded_imgs   = padded_imgs.to(device)   # [B, T, C, H, W]
            padded_labels = padded_labels.to(device) # [B, T, …]
            lengths       = lengths.to(device)       # [B]

            total_chunks = 0

            # iterate over each chunk of the sequence
            T = padded_imgs.size(1)
            total_chunks = T // chunk_size + (T % chunk_size > 0)
            for t0 in range(0, T, chunk_size):
                t1 = min(t0 + chunk_size, T)
                imgs_chunk   = padded_imgs[:, t0:t1]   # [B, chunk, C, H, W]
                labels_chunk = padded_labels[:, t0:t1] # [B, chunk, …]
                # adjust lengths for this chunk
                # mask out frames ≥ original length
                frame_idx = torch.arange(t0, t1, device=device).unsqueeze(0)  # [1, chunk]
                valid_mask = frame_idx < lengths.unsqueeze(1)                  # [B, chunk]

                # forward pass on just this chunk
                outputs = model((imgs_chunk, lengths), num_steps_per_image=num_steps)

                if getattr(model, "weighted_avg", False) is True:
                    outputs = logits_to_weighted_avg(outputs)
                for i, (output, max_value) in enumerate(zip(outputs, max_values)):
                    labels_valid = labels_chunk[:, :, i][valid_mask]
                    output = output.transpose(2, 1)
                    output_valid = output[valid_mask] * max_value
                    errors = torch.abs(output_valid.squeeze(-1) - labels_valid)
                    all_errors.extend(errors.cpu().numpy())
                del output, labels_valid, output_valid, max_value, errors, imgs_chunk, labels_chunk, outputs, valid_mask

            torch.cuda.empty_cache()
        if operation == "mean":
            avg_error_all_labels = []
            for i, label in enumerate(testloader.dataset.labels):
                avg_error = np.mean(all_errors)
                avg_error_all_labels.append(avg_error)
                if print_results: print(f"Average Error for {label}: {avg_error:.4f} pixels")
        elif operation == "distribution":
            plot_error_distribution(all_errors_x, all_errors_y)
        return np.array(avg_error_all_labels)

def evaluate_video_regression_tracker_OLD(model, testloader, device, max_values, num_steps=10, print_results=True, operation="mean", chunk_size=40):
    model.eval()  # Set model to evaluation mode

    all_errors = np.array([])

    with torch.no_grad():
        for padded_imgs, padded_labels, lengths in testloader:
            # Move everything to device once
            padded_imgs   = padded_imgs.to(device)   # [B, T, C, H, W]
            padded_labels = padded_labels.to(device) # [B, T, …]
            lengths       = lengths.to(device)       # [B]

            total_chunks = 0

            # iterate over each chunk of the sequence
            T = padded_imgs.size(1)
            total_chunks = T // chunk_size + (T % chunk_size > 0)
            for t0 in range(0, T, chunk_size):
                t1 = min(t0 + chunk_size, T)
                imgs_chunk   = padded_imgs[:, t0:t1]   # [B, chunk, C, H, W]
                labels_chunk = padded_labels[:, t0:t1] # [B, chunk, …]
                # adjust lengths for this chunk
                # mask out frames ≥ original length
                frame_idx = torch.arange(t0, t1, device=device).unsqueeze(0)  # [1, chunk]
                valid_mask = frame_idx < lengths.unsqueeze(1)                  # [B, chunk]

                # forward pass on just this chunk
                logits = model((imgs_chunk, lengths), num_steps_per_image=num_steps)

                total_loss = 0
                if getattr(model, "weighted_avg", False) is True:
                        outputs = logits_to_weighted_avg(outputs)
                for i, (output, max_value) in enumerate(zip(outputs, max_values)):
                    labels_valid = labels[mask][:, i]
                    output = output.transpose(2, 1)
                    output_valid = output[mask] * max_value
                    errors = torch.abs(output_valid.squeeze(-1) - labels_valid)
                    all_errors = np.concatenate((all_errors, errors.cpu().numpy()))
                del output, labels_valid, output_valid, loss, max_value


                # Accumulate grads over chunks and step once after the inner loop:
                total_chunks += 1

                # free up memory
                del imgs_chunk, labels_chunk, logits, loss, valid_mask

            torch.cuda.empty_cache()
        if operation == "mean":
            avg_error_all_labels = np.array([])
            for i, label in enumerate(testloader.dataset.labels):
                avg_error = np.mean(all_errors[i])
                avg_error_all_labels = np.concatenate((avg_error_all_labels, avg_error))
                if print_results: print(f"Average Error for {label}: {avg_error:.4f} pixels")
            return avg_error_all_labels
        elif operation == "distribution":
            plot_error_distribution(all_errors_x, all_errors_y)
            return avg_error_all_labels




        for padded_imgs, padded_labels, lengths in testloader:
            # Sort the batch by sequence lengths in descending order
            lengths, sorted_indices = lengths.sort(descending=True)
            lengths = lengths.to(device)
            padded_imgs = padded_imgs[sorted_indices].to(device)
            width, height = model.image_shape[2], model.image_shape[1]
            padded_labels = padded_labels[sorted_indices]
            # Use only x and y coordinates for classification.
            labels = padded_labels.round().long().to(device)
            
            outputs = model((padded_imgs, lengths), num_steps_per_image=num_steps)
            
            if hasattr(model, "weighted_avg") and model.weighted_avg:
                outputs = logits_to_weighted_avg(outputs)

            batch, num_classes_x, max_seq_length = outputs[0].shape

            # Create a mask: for each sequence in the batch, valid if t < length
            # The mask will have shape [batch, max_seq_length]
            mask = torch.arange(max_seq_length, device=outputs[0].device).expand(batch, max_seq_length) < lengths.unsqueeze(1)              # [batch * max_seq_length]
            avg_error_all_labels = np.empty((0, 0, 0))
            for i, (output, max_value) in enumerate(zip(outputs, max_values)):
                labels_valid = labels[mask][:, i]
                output = output.transpose(2, 1)
                output_valid = output[mask] * max_value
                total_loss += loss

            # weighted_x = weighted_x.transpose(2, 1)
            # weighted_y = weighted_y.transpose(2, 1)
            preds_x = weighted_x[mask] *  width
            preds_y = weighted_y[mask] * height
            
            # Compute absolute error per sample
            error_x = torch.abs(preds_x.float() - labels_x_valid.float())
            error_y = torch.abs(preds_y.float() - labels_y_valid.float())

            all_errors_x = np.concatenate((all_errors_x, error_x.cpu().numpy()))
            all_errors_y = np.concatenate((all_errors_y, error_y.cpu().numpy()))
        if operation == "mean":
            avg_error_x = np.mean(all_errors_x)
            avg_error_y = np.mean(all_errors_y)
            if print_results: print(f"Average X Error: {avg_error_x:.4f} pixels, Average Y Error: {avg_error_y:.4f} pixels")
            return avg_error_x, avg_error_y
        elif operation == "distribution":
            plot_error_distribution(all_errors_x, all_errors_y)
            return all_errors_x, all_errors_y
 

def plot_timestep_curve(model, testloader, device, identifier="", regression=False, interval = [1, 100, 5]):
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
                path = f'models/{model.model_type}WAvg_q{model.training_params["quantization"]}_{model.training_params["num_steps"]}ts_{model.training_params["num_epochs"]}e.pt'
            else:
                path = f'models/{model.model_type}_q{model.training_params["quantization"]}_{model.training_params["num_steps"]}ts_{model.training_params["num_epochs"]}e.pt'
        else: path = f'models/{model.model_type}_q{model.training_params["quantization"]}_{model.training_params["num_steps"]}ts_{model.training_params["num_epochs"]}e.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_params': model.training_params,
    }, path)
    print(f"Model saved at {path}")

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

def measure_inference_time_per_image_for_videos(model, dataloader, device, num_steps=10, num_batches=10):
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

def load_model(path, model_class, trainset, device):
    checkpoint = torch.load(path)
    # Create model instance with the same parameters as the saved model
    if checkpoint['training_params'].keys().__contains__('weighted_avg'):
        model = model_class(trainset, weighted_avg=True)
    else:
        model = model_class(trainset, weighted_avg=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.training_params = checkpoint['training_params']
    model.to(device)
    return model