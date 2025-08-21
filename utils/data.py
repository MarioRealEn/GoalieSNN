
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import cv2
from torch.utils.data import Dataset
import snntorch as snn
import torchvision.transforms.functional as TF
import random
import math
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import time
# import kornia.filters as kfilters


DATASET_PREFIX_MAP = {
            "all": [],
            "ball": ["13-38-07", "13-41-38"],
            "background": ["13-47-09"],
            "lemon": ["13-45-16"]
        }

ORIGINAL_IMAGE_SHAPE = (1280, 720, 2)  # Original image shape (height, width, channels)
MAX_VAL_R_CAM = 800
MAX_VAL_Y_CAM = 1854
MAX_VAL_Y_CAM = 720

def get_chosen_indices_frames(data, split, train_ratio, val_ratio, seed, column_name, dataset_type = None):
    if dataset_type is not None:    
        valid_prefixes = DATASET_PREFIX_MAP.get(dataset_type, [])
        if len(valid_prefixes) == 0:
            data = data.reset_index(drop=True)
        else:
            mask = data[column_name].apply(
                lambda f: any(str(f).startswith(pref) for pref in valid_prefixes)
            )
            data = data[mask].reset_index(drop=True)
            print(f"Filtered to {len(data)} rows for dataset type: {dataset_type}")

    total_size = len(data)
    indices = torch.randperm(total_size, generator=torch.Generator().manual_seed(seed))
    if split == "all":
        chosen_indices = indices
        return chosen_indices
    train_end = int(train_ratio * total_size)
    val_end = train_end + int(val_ratio * total_size)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    print(f"Split sizes: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}, total={total_size}")
    assert len(train_indices) + len(val_indices) + len(test_indices) == total_size, \
        "Split indices do not add up to total size"

    if split == "train":
        chosen_indices = train_indices
    elif split == "val":
        chosen_indices = val_indices
    elif split == "test":
        chosen_indices = test_indices
    else:
        raise ValueError("split must be 'train', 'val', 'test' or 'all'")
    return data, chosen_indices

def get_chosen_indices_videos(data, split, train_ratio, val_ratio, seed, column_name, dataset_type = "all", goal_type = "all"):
    if dataset_type == 'throws':
        data = data[data['is_roll'] == False].reset_index(drop=True)
    elif dataset_type == 'rolls':
        data = data[data['is_roll'] == True].reset_index(drop=True)
    elif dataset_type == 'all':
        pass
    else:
        raise ValueError("dataset_type must be 'throws', 'rolls' or 'all'")
    
    if goal_type == 'all':
        pass
    elif goal_type == 'out':
        data = data[data['type'] == 'out'].reset_index(drop=True)
    elif goal_type == 'in':
        data = data[data['type'] == 'in'].reset_index(drop=True)
    elif goal_type == 'almost_in':
        data = data[data['type'] == 'almost_in'].reset_index(drop=True)
    elif goal_type == 'towards_goal':
        data = data[(data['type'] == 'in') | (data['type'] == 'almost_in')].reset_index(drop=True)
    else:
        raise ValueError("goal_type must be 'all', 'out', 'in', 'almost_in' or 'towards_goal'")

    total_size = len(data)
    indices = torch.randperm(total_size, generator=torch.Generator().manual_seed(seed))
    if split == "all":
        chosen_indices = indices
        return data, chosen_indices
    train_end = int(train_ratio * total_size)
    val_end = train_end + int(val_ratio * total_size)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    print(f"Split sizes: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}, total={total_size}")
    assert len(train_indices) + len(val_indices) + len(test_indices) == total_size, \
        "Split indices do not add up to total size"

    if split == "train":
        chosen_indices = train_indices
    elif split == "val":
        chosen_indices = val_indices
    elif split == "test":
        chosen_indices = test_indices
    else:
        raise ValueError("split must be 'train', 'val', 'test' or 'all'")
    return data, chosen_indices

class BallTrackingDatasetImages(Dataset):
    def __init__(
        self,
        csv_path,
        image_dir,
        dataset_type="all",
        split="train",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        augment=False,
        vert_flip=False,
        rotation=False,
        seed=42,
        quantization=1,
        label_quantization=1,
        crop_margins=False,
        labels = ["x_cam", "y_cam", "R_cam"]
    ):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Train, val, and test ratios must sum to 1."
        self.split = split  # store so we know if we're in train/val/test
        self.augment = augment
        self.vert_flip = vert_flip
        self.rotation = rotation
        self.image_dir = image_dir
        self.quantization = quantization
        self.crop_margins = crop_margins
        self.labels = labels
        if quantization < 1 or label_quantization < 1:
            raise ValueError("Quantization factors must be >= 1")
        if quantization > 1 and label_quantization == 1:
            self.label_quantization = quantization
        else:
            self.label_quantization = label_quantization

        self.data = pd.read_csv(csv_path)
        print(f"Loaded {len(self.data)} rows from {csv_path}")

        self.data, chosen_indices = get_chosen_indices_frames(self.data, split, train_ratio, val_ratio, seed, column_name="Frame", dataset_type=dataset_type)

        self.data = self.data.iloc[chosen_indices].reset_index(drop=True)
        print(f"Final dataset split='{split}' size: {len(self.data)}")

        self.label_shape = None
        img, _ = self.__getitem__(0)  # Test the first item to check for errors
        self.image_shape = img.shape
        print(f"Shape of the images: {self.image_shape}")
        print(f"Label shape: {self.label_shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        frame_name = row["Frame"]
        labels = np.array([row[label] for label in self.labels])
        image_path = os.path.join(self.image_dir, str(frame_name) + ".png")
        image = cv2.imread(image_path)

        if self.crop_margins: image, labels = self.crop_item(image, labels)
        # Edit the coordinates to match the cropping
        x, y, r = labels // self.label_quantization
        
        width, height = image.shape[1], image.shape[0]

        # Quantize the image
        width_img, height_img = width // self.quantization, height // self.quantization
        width_label, height_label = width // self.label_quantization, height // self.label_quantization
        if self.label_shape is None:
            self.label_shape = (width_label, height_label)
        image_tensor = TF.to_tensor(image)
        if self.quantization > 1:
            image_tensor = F.interpolate(image_tensor.unsqueeze(0), size=(height_img, width_img), mode='bilinear', align_corners=False)
            image_tensor = image_tensor.squeeze(0)
        image_tensor = image_tensor[[0, 2]]

        if pd.isna(x) or pd.isna(y) or pd.isna(r):
            label = torch.tensor([-1, -1, -1], dtype=torch.float32)
            print(f"Warning: Missing label for {frame_name}")
        else:
            if self.augment:
                if random.random() > 0.5:
                    image_tensor = TF.hflip(image_tensor)
                    x = width_label - x

                if random.random() > 0.5 and self.vert_flip:
                    image_tensor = TF.vflip(image_tensor)
                    y = height_label - y

                if random.random() > 0.5 and self.rotation:
                    angle = random.uniform(-15, 15)
                    image_tensor = TF.rotate(image_tensor, angle)

                    cx, cy = width_label / 2, height_label / 2
                    angle_rad = math.radians(angle)
                    x_new = math.cos(angle_rad) * (x - cx) - math.sin(angle_rad) * (y - cy) + cx
                    y_new = math.sin(angle_rad) * (x - cx) + math.cos(angle_rad) * (y - cy) + cy
                    x, y = x_new, y_new
            if x < 0 or y < 0 or x >= width_label or y >= height_label:
                print(f"Warning: Label out of bounds for {frame_name}, x={x}, y={y}")
                x = min(width_label, max(0, x))
                y = min(height_label, max(0, y))
                label = torch.tensor([x, y, r], dtype=torch.float32)
            else:
                label = torch.tensor([x, y, r], dtype=torch.float32)
        return image_tensor, label

    def crop_item(self, image, labels):
        cropped_image, crop_coords = self.crop_white_margins(image)
        new_labels = self.crop_label(labels, crop_coords)
        return cropped_image, new_labels
    
    def crop_white_margins(self, image):
        """
        Automatically crops white margins from an image.
        Works for both grayscale and colored images.
        """
        # # Convert to grayscale if the image is colored
        # if len(image.shape) == 3 and image.shape[2] == 3:
        #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # else:
        #     gray = image.copy()
        
        # # Threshold the grayscale image: assume near-white areas (>=240) are background
        # _, bin_img = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        
        # # Find all non-white pixels (invert the binary image)
        # coords = cv2.findNonZero(255 - bin_img)
        
        # # Get bounding box of non-white region
        # x, y, w, h = cv2.boundingRect(coords)
        
        # # Crop the original image using the bounding box
        # cropped_img = image[y:y+h, x:x+w]
        
        # HARDCODED VERSION
        margin = 30
        y = 128 - margin
        h = 348 + 2*margin
        x = 100 - margin
        w = 620 + 2*margin
        cropped_img = image[y:y+h, x:x+w]
        # Make the values on the margin black
        cropped_img[:margin+1, :] = 0
        cropped_img[-margin:, :] = 0
        cropped_img[:, :margin+1] = 0
        cropped_img[:, -margin:] = 0

        return np.array(cropped_img), [x, y, w, h]

    def crop_label(self, label, crop_coords):
        x, y, r = label
        x -= crop_coords[0]
        y -= crop_coords[1]
        return np.array([x, y, r])
    
    def create_cropped_dataset(self, folder_path):
        """
        Create a new dataset with cropped images and labels.
        """
        new_df = pd.DataFrame(columns=["Frame", "x_cam", "y_cam", "R_cam"])
        for idx in range(len(self)):
            row = self.data.iloc[idx]
            frame_name, x, y, r = row["Frame"], row["x_cam"], row["y_cam"], row["R_cam"]
            image_path = os.path.join(self.image_dir, str(frame_name) + ".png")
            # image = Image.open(image_path)
            image = cv2.imread(image_path)
            # Convert to np array for OpenCV
            image = np.array(image)

            cropped_image, labels = self.crop_item(image, [x, y, r])
            # Save the cropped image
            cropped_image_folder = os.path.join(folder_path, 'frames')
            os.makedirs(cropped_image_folder, exist_ok=True)
            # Save the cropped image
            cropped_image_path = os.path.join(cropped_image_folder, str(frame_name) + ".png")
            cv2.imwrite(cropped_image_path, cropped_image)
            new_row = {
                "Frame": frame_name,
                "x_cam": labels[0],
                "y_cam": labels[1],
                "R_cam": labels[2]
            }
            new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
        new_df = new_df.sort_values(by=["Frame"])
        new_df = new_df.reset_index(drop=True)
        new_df.to_csv(os.path.join(folder_path, 'labels.csv'), index=False)
        print(f"Cropped dataset saved to {folder_path}")


class BallTrackingDatasetVarLenVideos(Dataset):
    def __init__(
        self,
        csv_path,
        slices_dir,
        dataset_type="all",
        split="train",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        augment=False,
        vert_flip=False,
        rotation=False,
        seed=42,
        quantization=1,
        label_quantization=1,
        crop_margins=False,
        slice_type="frame", # Type 'video' is not implemented
        labels = ["x_cam", "y_cam", "R_cam"]
    ):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Train, val, and test ratios must sum to 1."
        self.split = split
        self.augment = augment
        self.vert_flip = vert_flip
        self.rotation = rotation
        self.slices_dir = slices_dir
        self.quantization = quantization
        self.crop_margins = crop_margins
        self.slice_type = slice_type
        self.labels = labels
        self.n_frames = None
        if quantization < 1 or label_quantization < 1:
            raise ValueError("Quantization factors must be >= 1")
        if quantization > 1 and label_quantization == 1:
            self.label_quantization = quantization
        else:
            self.label_quantization = label_quantization

        self.frames= pd.read_csv(csv_path)
        print(f"Loaded {len(self.frames)} rows from {csv_path}")
        data = self.frames['Frame'].str.split('_').str[:2].str.join('_').unique()
        self.data = pd.DataFrame(data, columns=['VideoName'])
        self.frames['VideoName'] = self.frames['Frame'].str.split('_').str[:2].str.join('_')
        print(f"Found {len(self.data)} sequences in {csv_path}")

        self.label_shape = None
        video, _, _ = self.__getitem__(0)  # Test the first item to check for errors
        img = video[0]
        self.image_shape = img.shape
        print(f"Shape of the images: {self.image_shape}")
        print(f"Label shape: {self.label_shape}")

        self.data, chosen_indices = get_chosen_indices_frames(self.data, split, train_ratio, val_ratio, seed, column_name="VideoName", dataset_type=dataset_type)

        self.data = self.data.iloc[chosen_indices]
        print(f"Final dataset split='{split}' size: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        trajectory = self.data.iloc[idx]["VideoName"] if self.slice_type == "frame" else self.data.iloc[idx]["tr"]
        frames_video = self.frames[self.frames["VideoName"] == trajectory] if self.slice_type == "frame" else self.frames[self.frames["tr"] == trajectory]
        images = []
        labels = []
        transformation = None
        if self.split == "train" and self.augment:
            if random.random() > 0.5:
                transformation = "hflip"

            if random.random() > 0.5 and self.vert_flip:
                transformation = "vflip"

            if random.random() > 0.5 and self.rotation:
                transformation = "rotate"
                angle = random.uniform(-15, 15)
        if self.slice_type == "video":
            video_file = os.path.join(self.slices_dir, f"tr{str(trajectory)}" + ".mp4")
            video = cv2.VideoCapture(video_file)
            if not video.isOpened():
                raise ValueError(f"Could not open video file: {video_file}")
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Video {video_file} has {frame_count} frames")
            print(f"Frames video: {frames_video}")

        for _, row in frames_video.iterrows():
            # print(row)
            frame_name = row["Frame"]
            label = np.array([row[label] for label in self.labels])
            if self.slice_type == "frame":
                image_path = os.path.join(self.slices_dir, str(frame_name) + ".png")
                image = cv2.imread(image_path)
            elif self.slice_type == "video":
                image_path = os.path.join(self.slices_dir, str(trajectory) + ".png")
                image = cv2.imread(image_path)

            if self.crop_margins: image, label = self.crop_item(image, label)
            x, y, r = label // self.label_quantization

            width, height = image.shape[1], image.shape[0]

            # Quantize the image
            width_img, height_img = width // self.quantization, height // self.quantization
            width_label, height_label = width // self.label_quantization, height // self.label_quantization
            if self.label_shape is None:
                self.label_shape = (width_label, height_label)
            image_tensor = TF.to_tensor(image)
            if self.quantization > 1:
                image_tensor = F.interpolate(image_tensor.unsqueeze(0), size=(height_img, width_img), mode='bilinear', align_corners=False)
                image_tensor = image_tensor.squeeze(0)
            image_tensor = image_tensor[[0, 2]]

            if pd.isna(x) or pd.isna(y) or pd.isna(r):
                label = torch.tensor([-1, -1, -1], dtype=torch.float32)
                print(f"Warning: Missing label for {frame_name}")
            else:
                if transformation == "hflip":
                    image_tensor = TF.hflip(image_tensor)
                    x = width - x

                elif transformation == "vflip":
                    image_tensor = TF.vflip(image_tensor)
                    y = height - y

                elif transformation == "rotate":
                    image_tensor = TF.rotate(image_tensor, angle)
                    cx, cy = width / 2, height / 2
                    angle_rad = math.radians(angle)
                    x_new = math.cos(angle_rad) * (x - cx) - math.sin(angle_rad) * (y - cy) + cx
                    y_new = math.sin(angle_rad) * (x - cx) + math.cos(angle_rad) * (y - cy) + cy
                    x, y = x_new, y_new
                elif transformation is None:
                    pass
                else:
                    raise ValueError("Invalid transformation")
            
                if x < 0 or y < 0 or x >= width or y >= height:
                    print(f"Warning: Label out of bounds for {frame_name}, x={x}, y={y}")
                    x = min(width, max(0, x))
                    y = min(height, max(0, y))
                    label = torch.tensor([x, y, r], dtype=torch.float32)
                else:
                    label = torch.tensor([x, y, r], dtype=torch.float32)
            images.append(image_tensor)
            labels.append(label)
            
        images = torch.stack(images)
        labels = torch.stack(labels)
            
        return images, labels, images.shape[0]

    def crop_item(self, image, labels):
        cropped_image, crop_coords = self.crop_white_margins(image)
        new_labels = self.crop_label(labels, crop_coords)
        return cropped_image, new_labels

    def crop_white_margins(self, image):
        """
        Automatically crops white margins from an image.
        Works for both grayscale and colored images.
        """
        # DYNAMIC VERSION it didnt work well, the output images had different sizes and sometimes white lines
        # # Convert to grayscale if the image is colored
        # if len(image.shape) == 3 and image.shape[2] == 3:
        #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # else:
        #     gray = image.copy()
        
        # # Threshold the grayscale image: assume near-white areas (>=240) are background
        # _, bin_img = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        
        # # Find all non-white pixels (invert the binary image)
        # coords = cv2.findNonZero(255 - bin_img)
        
        # # Get bounding box of non-white region
        # x, y, w, h = cv2.boundingRect(coords)
        
        # # Crop the original image using the bounding box
        # cropped_img = image[y:y+h, x:x+w]
        
        # HARDCODED VERSION
        margin = 30
        y = 128 - margin
        h = 348 + 2*margin
        x = 100 - margin
        w = 620 + 2*margin
        cropped_img = image[y:y+h, x:x+w]
        # Make the values on the margin black
        cropped_img[:margin+1, :] = 0
        cropped_img[-margin:, :] = 0
        cropped_img[:, :margin+1] = 0
        cropped_img[:, -margin:] = 0

        return np.array(cropped_img), [x, y, w, h]

    def crop_label(self, label, crop_coords):
        x, y, r = label
        x -= crop_coords[0]
        y -= crop_coords[1]
        return np.array([x, y, r])
    
    def return_n_frames(self):
        if self.n_frames is None:
            n_frames = 0
            for _, video in self.data.iterrows():
                vid = video['VideoName']
                pos_tr = self.frames[self.frames['VideoName'] == vid]
                n_frames += len(pos_tr)
            self.n_frames = n_frames
        return self.n_frames
    
    # Custom collate function for padding sequences in a batch. This is important for variable-length sequences.
    @staticmethod
    def collate_fn(batch):
        # Each item in batch is a tuple: (imgs, labels, length)
        imgs_list, labels_list, lengths = zip(*batch)
        # Pad image sequences; assuming images are tensors of shape [seq_length, ...]
        padded_imgs = pad_sequence(imgs_list, batch_first=True)
        # Pad labels; here we assume labels are 1D tensors. Adjust dim if needed.
        padded_labels = pad_sequence(labels_list, batch_first=True, padding_value=-1)
        lengths = torch.tensor(lengths)
        return padded_imgs, padded_labels, lengths

class Tracking3DVideoDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        accumulation_time,
        positions_csv = None,
        dataset_type="all",
        goal_type="all",
        split="train",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        augment=False,
        vert_flip=False,
        rotation=False,
        seed=42,
        quantization=1,
        label_quantization=None,
        crop_margins=False,
        labels = ["x_cam", "y_cam", "R_cam"]
    ):
        print(f"Loading dataset from {dataset_path} with accumulation time {accumulation_time}ms")
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Train, val, and test ratios must sum to 1."
        self.split = split
        self.augment = augment
        self.vert_flip = vert_flip
        self.rotation = rotation
        self.slices_dir = os.path.join(dataset_path, f'avi_videos_{accumulation_time}ms')
        if positions_csv == None: csv_path = os.path.join(dataset_path, f'positions_{accumulation_time}ms.csv') 
        else: csv_path = positions_csv
        self.quantization = quantization
        self.crop_margins = crop_margins
        self.labels = labels
        self.n_frames = None
        self._video_cache = {}
        self._label_cache = {}
        if quantization >= 1 and label_quantization is None:
            self.label_quantization = quantization
        else:
            self.label_quantization = label_quantization
        if quantization < 1 or self.label_quantization < 1:
            raise ValueError("Quantization factors must be >= 1")

        self.positions= pd.read_csv(csv_path)
        print(f"Loaded {len(self.positions)} rows from {csv_path}")
        trajectories = self.positions['tr'].unique()
        trajectories_csv = os.path.join(dataset_path, f'trajectories.csv')
        self.trajectories = pd.read_csv(trajectories_csv)
        self.trajectories = self.trajectories[self.trajectories['tr'].isin(trajectories)]
        print(f"Found {len(self.trajectories)} sequences in {csv_path}")

        video, labels, _ = self.__getitem__(0)  # Test the first item to check for errors
        img = video[0]
        self.image_shape = img.shape
        self.n_fields = labels.shape[-1]
        print(f"Shape of the images: {self.image_shape}")
        print(f"Number of label fields: {self.n_fields}")

        self.trajectories, chosen_indices = get_chosen_indices_videos(self.trajectories, split, train_ratio, val_ratio, seed, column_name="tr", dataset_type=dataset_type, goal_type=goal_type)

        self.trajectories = self.trajectories.iloc[chosen_indices].reset_index(drop=True)
        print(f"Final dataset split='{split}' size: {len(self.trajectories)}\n")


    def __len__(self):
        return len(self.trajectories)
    
    def return_n_frames(self):
        if self.n_frames is None:
            n_frames = 0
            for _, trajectory in self.trajectories.iterrows():
                tr = trajectory['tr']
                pos_tr = self.positions[self.positions['tr'] == tr]
                n_frames += len(pos_tr)
            self.n_frames = n_frames
        return self.n_frames

    def crop_item(self, image, labels):
        cropped_image, crop_coords = self.crop_white_margins(image)
        new_labels = self.crop_label(labels, crop_coords)
        return cropped_image, new_labels

    def crop_white_margins(self, image):
        """
        Automatically crops white margins from an image.
        Works for both grayscale and colored images.
        """
        # HARDCODED VERSION
        margin = 30
        y = 128 - margin
        h = 348 + 2*margin
        x = 100 - margin
        w = 620 + 2*margin
        cropped_img = image[y:y+h, x:x+w]
        # Make the values on the margin black
        cropped_img[:margin+1, :] = 0
        cropped_img[-margin:, :] = 0
        cropped_img[:, :margin+1] = 0
        cropped_img[:, -margin:] = 0

        return np.array(cropped_img), [x, y, w, h]

    def crop_label(self, label, crop_coords):
        x, y, r = label
        x -= crop_coords[0]
        y -= crop_coords[1]
        return np.array([x, y, r])
    
    # Custom collate function for padding sequences in a batch. This is important for variable-length sequences.
    @staticmethod
    def collate_fn(batch):
        # Each item in batch is a tuple: (imgs, labels, length)
        imgs_list, labels_list, lengths = zip(*batch)
        # Pad image sequences; assuming images are tensors of shape [seq_length, ...]
        padded_imgs = pad_sequence(imgs_list, batch_first=True)
        # Pad labels; here we assume labels are 1D tensors. Adjust dim if needed.
        padded_labels = pad_sequence(labels_list, batch_first=True, padding_value=-1)
        lengths = torch.tensor(lengths)
        return padded_imgs, padded_labels, lengths
    
    def clear_cache(self):
        """Call this at the end of each epoch to free all cached videos."""
        self._video_cache.clear()

    def _load_full_video(self, trajectory_id):
        """Loads and caches the full video tensor for a given trajectory."""
        path = os.path.join(self.slices_dir, f"tr{trajectory_id}.avi")
        cap  = cv2.VideoCapture(path) #, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # keep channels 0 and 2
            frames.append(frame[..., [0, 2]])
        cap.release()

        # Convert to tensor [T, 2, H, W]
        arr = np.stack(frames, axis=0)           # [T, H, W, 2]
        t   = torch.from_numpy(arr).float()/255      # [T, H, W, 2]
        return t.permute(0, 3, 1, 2)             # [T, 2, H, W]

    def __getitem__(self, idx):
        # Determine trajectory
        trajectory = self.trajectories.iloc[idx]["tr"]
        video, labels, length = self.__getitemtr__(trajectory)  # Ensure the trajectory is loaded
        return video, labels, length

    def __getitemtr__(self, trajectory):
        frames_video = self.positions[self.positions["tr"] == trajectory]
        if frames_video.empty:
            raise IndexError(f"No frames for trajectory {trajectory}")
        video_from_cache = True
        # 2) Load or fetch from cache the full video tensor
        if trajectory not in self._video_cache:
            full_video = self._load_full_video(trajectory)
            video_from_cache = False
            T, C, H, W = full_video.shape
            H, W = H // self.quantization, W // self.quantization
        else:
            full_video = self._video_cache[trajectory]  # [T, 2, H, W]
            T, C, H, W = full_video.shape
            labels_cache = self._label_cache[trajectory]  # [T, N_labels]

        # 3) Decide augmentation once per sequence
        transformation = None
        angle = 0.0
        if self.augment:
            if random.random() > 0.5:
                transformation = "hflip"
            if random.random() > 0.5 and self.vert_flip:
                transformation = "vflip"
            if random.random() > 0.5 and self.rotation:
                transformation = "rotate"
                angle = random.uniform(-15, 15)
        video = []
        labels = []

        # 4) Iterate only over the rows you need
        in_fov_flag = False
        in_fov = None
        label_fields = self.labels.copy()
        if 'in_fov' in label_fields:
            in_fov_flag = True
            label_fields.remove('in_fov')
        for _, row in frames_video.iterrows():
            n_frame = int(row["frame"])
            if n_frame < 0 or n_frame >= T:
                continue  # skip out-of-bounds

            # slice single frame [2, H, W]
            image_tensor = full_video[n_frame]

            # fetch and quantize label
            if video_from_cache:
                if in_fov_flag:
                    x, y, r, in_fov = labels_cache[n_frame]
                else:
                    x, y, r = labels_cache[n_frame]
            else:
                raw = np.array([row[l] for l in label_fields])
                x, y, r = raw / self.label_quantization
                r = min(max(0, r), MAX_VAL_R_CAM//self.label_quantization)  # clamp to [0, MAX_VAL_R_CAM]
                y = min(max(0, y), MAX_VAL_Y_CAM//self.label_quantization)  # clamp to [0, MAX_VAL_Y_CAM]
                # if r > 100:
                #     print(f"Warning: R value out of bounds for {row['Frame']}, r={r}")
                #     print(raw)
                #     print(self.label_quantization)
                #     print(MAX_VAL_R_CAM)
                if in_fov_flag: in_fov = row["in_fov"]

                # apply quantization to image
                if self.quantization > 1:
                    # h_q = H
                    # w_q = W
                    # image_tensor = F.interpolate(
                    #     image_tensor.unsqueeze(0), 
                    #     size=(h_q, w_q), 
                    #     mode="bilinear", 
                    #     align_corners=False
                    # ).squeeze(0)

                    image_tensor = F.max_pool2d(
                        image_tensor.unsqueeze(0),            # add batch dim
                        kernel_size=self.quantization,
                        stride=self.quantization
                    ).squeeze(0)

                    # image_tensor = cv2.pyrDown(image_tensor.permute(1, 2, 0).numpy(),
                    #                            dstsize=(W*2, H*2))
                    # image_tensor = cv2.pyrDown(image_tensor, dstsize=(W, H))

                    # image_tensor = torch.from_numpy(image_tensor).float()

                    # image_tensor = image_tensor.permute(2, 0, 1)  # [H, W, C] to [C, H, W]

                    # image_tensor = self.bilateral_downsample(image_tensor, scale=self.quantization)

            # 5) Data augmentation on this single frame
            if transformation == "hflip":
                x = W - x
            elif transformation == "vflip":
                y = H - y
            elif transformation == "rotate":
                cx, cy = W / 2, H / 2
                ar = math.radians(angle)
                x_new = math.cos(ar) * (x - cx) - math.sin(ar) * (y - cy) + cx
                y_new = math.sin(ar) * (x - cx) + math.cos(ar) * (y - cy) + cy
                x, y = x_new, y_new

            # 6) Build final label tensor
            label_vals = [x, y, r, in_fov] if in_fov is not None else [x, y, r]
            label = torch.tensor(label_vals, dtype=torch.float32)

            video.append(image_tensor)
            labels.append(label)
        video = torch.stack(video, dim=0)  # [N_valid, 2, H', W']
        if transformation == "hflip":
            video = TF.hflip(video)
        elif transformation == "vflip":
            video = TF.vflip(video)
        elif transformation == "rotate":
            video = TF.rotate(video, angle)
        labels = torch.stack(labels, dim=0)  # [N_valid, N_labels]
        if not video_from_cache:
            self._video_cache[trajectory] = video  # cache the video tensor
            self._label_cache[trajectory] = labels  # cache the label tensor
        return video, labels, video.size(0)

    def __gettr__(self, idx):
        """Get the trajectory ID for a given index."""
        return self.trajectories.iloc[idx]["tr"]
    
    def __getidx__(self, trajectory):
        """Get the index of a trajectory in the dataset."""
        idx = self.trajectories[self.trajectories["tr"] == trajectory].index
        if len(idx) == 0:
            raise ValueError(f"Trajectory {trajectory} not found in dataset")
        return idx[0]
    
    def __getcachesize__(self):
        """Get the size of the video cache."""
        size = 0
        for key in self._video_cache:
            size += self.__getsizeinmbs__(self._video_cache[key])
        return size
    
    def __getsizeinmbs__(self, traj):
        return traj.numel() * traj.element_size() / (1024**2)  # in MB
    
    def downscale_gaussian(self, frame, scale):
        """Blurs then resizes for arbitrary downscaling."""
        # 1) Compute sigma for gaussian blur: a rule-of-thumb is sigma ≈ 0.5 * minKernelRadius
        sigma = 0.8
        # kernel size: odd, e.g. 5 or 7; larger means stronger denoising
        ksize = 5  
        blurred = cv2.GaussianBlur(frame, (ksize, ksize), sigmaX=sigma)
        # 2) Resize with INTER_AREA, which is best for downsampling
        h, w = frame.shape[:2]
        new_size = (int(w/scale), int(h/scale))
        return cv2.resize(blurred, new_size, interpolation=cv2.INTER_AREA)
    
    def gaussian_downsample(self,
                            x: torch.Tensor,
                            scale: int = 2,
                            kernel_size: int = 5,
                            sigma: float = 1.0) -> torch.Tensor:
        """
        Blur + downsample a [B,C,H,W] tensor by an integer `scale`.
        - Applies a Gaussian low-pass (conv2d) per channel
        - Then uses area-based interpolate for clean subsampling

        Args:
        x          : input tensor, shape [B,C,H,W]
        scale      : integer downsampling factor (e.g. 2 → halves H,W)
        kernel_size: odd size of the Gaussian kernel
        sigma      : standard deviation for Gaussian

        Returns:
        Tensor of shape [B,C,H/scale,W/scale]
        """
        C, H, W = x.shape
        # Build a single-channel 2D Gaussian kernel
        ax = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - (kernel_size - 1) / 2.
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        # Expand to (C×1×K×K) for group conv
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(C, 1, 1, 1)

        # 1) Blur each channel
        padding = kernel_size // 2
        x_blur = F.conv2d(x, kernel, groups=C, padding=padding)

        # 2) Downsample using area interpolation (best for decimation)
        new_h, new_w = H // scale, W // scale
        x_down = F.interpolate(x_blur.unsqueeze(0), size=(new_h, new_w),
                            mode='area').squeeze(0)
        return x_down
    
    def bilateral_downsample(self, x, scale=2,
                            kernel_size=(7,7), sigma_space=(2.0,2.0),
                            sigma_color=(0.1,0.1)):
        # x: [B,C,H,W], float in [0,1] or [0,255]
        # 1) edge-preserving blur
        x_blur = kfilters.bilateral_blur(
            x.unsqueeze(0), kernel_size=kernel_size,
            sigma_space=sigma_space,
            sigma_color=sigma_color
        ).squeeze(0)
        # 2) decimate with area interp
        new_h, new_w = x.shape[-2] // scale, x.shape[-1] // scale
        return F.interpolate(x_blur, size=(new_h,new_w), mode='area')



# GENERATORS

def prediction_generator_classification(model, testset, device, num_steps=20): # This one shows also the prediction from the model
    """Generator that yields images, labels, and predictions one at a time."""
    quantization_model = model.training_params["quantization"]
    quantization_testset = testset.quantization
    if quantization_model != quantization_testset:
        print(f"Quantization model: {quantization_model}, Quantization testset: {quantization_testset}")
        raise ValueError("Quantization factors of the model and the testset must be the same")
    for idx in range(len(testset)):  # Loop through dataset
        img, label = testset[idx]  # Get an item
        height, width = img[0].shape

        # Get model predictions
        probsx, probsy = model(img.unsqueeze(0).to(device), num_steps=num_steps)
        pred_x = torch.argmax(probsx, dim=1).item()
        pred_y = torch.argmax(probsy, dim=1).item()
        pred = (pred_x, pred_y)
        yield idx, img, label, pred, height, width

def prediction_generator_regression(model, testset, device, num_steps=20): # This one shows also the prediction from the model
    """Generator that yields images, labels, and predictions one at a time."""
    for idx in range(len(testset)):  # Loop through dataset
        img, label = testset[idx]  # Get an item
        height, width = img[0].shape

        # Get model predictions
        outputs = model(img.unsqueeze(0).to(device), num_steps=num_steps)
        preds = []
        for _, (output, max_val) in enumerate(zip(outputs, model.max_values)): # This is not right. Label fields should be an input to the function and be used to compute the max_values
            pred = output * max_val
            preds.append(pred.item())

        yield idx, img, label, preds, height, width

def image_generator(testset): # This one just shows the data
    """Generator that yields images, labels, and predictions one at a time."""
    for idx in range(len(testset)):  # Loop through dataset
        img, label = testset[idx]  # Get an item
        height, width = img[0].shape

        yield idx, img, label, height, width

def image_gen_for_video(video, labels, preds = None): # This one just shows the data
    """Generator that yields images, labels, and predictions given one sequence of images."""
    for idx in range(len(video)):  # Loop through dataset
        img = video[idx]  # Get an item
        label = labels[idx]
        height, width = img[0].shape

        if preds is None: yield idx, img, label, height, width
        else: yield idx, img, label, [preds[i, idx] for i in range(len(preds))], height, width

def increase_contrast(tensor_img, factor):
    """
    tensor_img: torch.Tensor, shape [C, H, W], values assumed in [0, 1]
    factor: float, e.g., 1.5 for +50% contrast
    """
    return torch.clamp((tensor_img - 0.5) * factor + 0.5, 0, 1)

# Function to fetch and display the next image
def show_next_img(gen, show_labels=True, quantization_factor=1, is_bgr=False, just_image=False):
    idx, img, label, height, width = next(gen)  # Get next sample from generator
    img = increase_contrast(img, 5) # Increase contrast
    print(height, width)
    print('max', img.max())
    print('min', img.min())
    print("Label: ", label)
    img_np = tensor_to_image(img)
    plt.figure(figsize=(20, 20))
    if is_bgr:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    plt.imshow(img_np, cmap='gray')
    if show_labels:
        # plt.scatter(label[0] * quantization_factor, height - label[1] * quantization_factor, c='r', label="Ground Truth")
        # Draw a yellow cross at the position of the label
        plt.scatter(label[0] * quantization_factor, height - label[1] * quantization_factor, c='y', marker='+', s=200, label="Label Position", linewidths=3)
        if len(label) > 2: # Draw a circle with radius pred[2] and center (pred[0], pred[1])
            true_circle = plt.Circle((label[0] * quantization_factor, (height - label[1]) * quantization_factor), label[2] * quantization_factor, color='yellow', fill=False, label="True Radius", linewidth=3)
            plt.gca().add_artist(true_circle)
    if just_image:
        plt.axis('off')
    else:
        if show_labels: plt.legend()
        plt.title(f"Sample {idx}")
    plt.show()

def show_next_img_separate(gen, quantization = 1, label_quantization = 1): # This is for when the image and the labels have different quantizations, so the axis are different
    idx, img, label, height, width = next(gen)  # Get next sample from generator
    print("Label: ", label)
    img_np = tensor_to_image(img)
    plt.figure()
    plt.imshow(img_np, cmap='gray')    
    plt.figure()
    plt.scatter(label[0], label[1], c='r', label="Ground Truth")
    max_x = width*quantization/label_quantization
    max_y = height*quantization/label_quantization
    print(max_x, max_y)
    plt.xlim(0, max_x)
    plt.ylim(0, max_y)
    plt.title(f"Sample {idx}")
    plt.legend()

def show_next_img_w_pred(gen):
    idx, img, label, pred, height, width = next(gen)  # Get next sample from generator
    img_np = tensor_to_image(img)
    plt.figure()
    plt.imshow(img_np, cmap='gray')
    plt.scatter(label[0], height - label[1], c='r', label="Ground Truth")
    plt.scatter(pred[0], height - pred[1], c='b', label="Prediction")
    if len(pred) > 2: # Draw a circle with radius pred[2] and center (pred[0], pred[1])
        true_circle = plt.Circle((label[0], height - label[1]), label[2], color='r', fill=False, label="True Radius")
        pred_circle = plt.Circle((pred[0], height - pred[1]), pred[2], color='b', fill=False, label="Pred Radius")
        plt.gca().add_artist(true_circle)
        plt.gca().add_artist(pred_circle)
    plt.title(f"Sample {idx}")
    plt.legend()
    plt.show()

def tensor_to_image(img):
    zero_channel = torch.zeros_like(img[0])
    img = torch.cat((img[0].unsqueeze(0), zero_channel.unsqueeze(0), img[1].unsqueeze(0)), dim=0)
    img_np = (img.cpu().numpy().transpose((1,2,0))*255).astype(np.uint8)
    return img_np