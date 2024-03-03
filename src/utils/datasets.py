import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import tifffile
import cv2
from os.path import join, isfile, exists
from PIL import Image
import re
from sklearn.model_selection import train_test_split
from torchvision import transforms
import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


label_map = {"BMP4" :0, "CHIR": 1, "DS": 2, "DS+CHIR": 3,  "WT": 4}
label_list = ["BMP4", "CHIR", "DS", "DS+CHIR",  "WT"]
# Define the input shape of the images
input_shape = (3, 224, 224)
input_shape_inception = (3, 299, 299)
# Define the number of classes
num_classes = 5

class Transforms:
    def __init__(self, rotation_degrees=[0, 90, 180, 270], input_shape=(3, 224, 224)):
        self.rotation_degrees = rotation_degrees
        self.input_shape = input_shape

    def rotate_image(self, image, degree):
        return transforms.functional.rotate(image, degree)

    def get_transforms(self, image):
        transformed_images = []
        for degree in self.rotation_degrees:
            img = self.rotate_image(image, degree)
            transformed_images.append(img)
        return transformed_images
def preprocess_tiffimage(image_array):
    img_rescaled = 255 * (image_array - image_array.min()) / (image_array.max() - image_array.min())
    img = Image.fromarray(img_rescaled)
    img = img.convert("RGB")
    return img

def proprocess_image(tensor):
    # Define the mean and standard deviation used for the initial normalization
    mean = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1).to(device)
    std = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1).to(device)
    # mean = torch.tensor([0.1169, 0.1169, 0.1169]).view(-1, 1, 1).to(device)
    # std = torch.tensor([0.0929, 0.0929, 0.0929]).view(-1, 1, 1).to(device)
    # Reverse the normalization process
    tensor = tensor * std + mean

    # Clip values to ensure they are within a valid range
    tensor = torch.clamp(tensor, min=0.0, max=1.0)

    return tensor
    
def get_condition(title):
    # Load the Excel file into a DataFrame
    df = pd.read_excel('../fig5_tile_conditions.xlsx')

    # Filter the DataFrame based on the given parameters
    filtered_df = df[(df['Experiment'].dt.day == int(title.split('_')[0])) & (df['Tile'] == int(title.split('_')[1]))]

    # Check if there are any matching rows
    if len(filtered_df) > 0:
        # Retrieve the condition from the first matching row
        condition = filtered_df.iloc[0]['Condition']
        return condition
    else:
        return None


# Define the dataset class
class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None, timestamp=1, data_enrich=False):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        image_transforms = Transforms(rotation_degrees=[0, 90, 180, 270])
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            files = [f for f in os.listdir(folder_path) if isfile(join(folder_path, f))]
            label = get_condition(folder)
            for file in files:
                file_path = os.path.join(folder_path, file)
                match = re.search(r'_t(\d+)_c002', file)
                if match:
                    number = int(match.group(1))
                    if number == timestamp:
                        if file_path.endswith(".png"):
                            img = Image.open(file_path).convert("RGB")
                            name = file.split("_")[0]
                            label = label_map.get(label, -1)
                            if label != -1:
                                self.images.append(img)
                                self.labels.append(label)
                        elif file_path.endswith(".tif"):                     
                            try:
                                image_array = tifffile.imread(file_path)
                            except TypeError:
                                pass
                            img = preprocess_tiffimage(image_array)
                            name = file.split("_")[0]
                            label = label_map.get(label, -1)
                            if label != -1:
                                if data_enrich:
                                    images = image_transforms.get_transforms(img)
                                    for image in images:
                                        self.images.append(image)
                                        self.labels.append(label)
                                else:
                                    self.images.append(img)
                                    self.labels.append(label)
                            break
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        # apply transformation         
        if self.transform is not None:
            img = self.transform(img)
        return img, label

transform = transforms.Compose([
    transforms.Resize(input_shape[1:]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_inception = transforms.Compose([
    transforms.Resize(input_shape_inception[1:]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the dataset
def get_dataloaders(time, test_ratio=0.2, inception=False, data_enrich=False):
    if inception:
        dataset = MyDataset("../../images", transform=transform_inception, timestamp=time, data_enrich=data_enrich)
    else:
        dataset = MyDataset("../../images", transform=transform, timestamp=time, data_enrich=data_enrich)
    labels = dataset.labels
    train_dataset, remaining_data, train_labels, remaining_labels = train_test_split(dataset, labels, test_size=test_ratio, stratify=labels, random_state=42)
    validation_dataset, test_dataset, validation_labels, test_labels = train_test_split(remaining_data, remaining_labels, test_size=0.5, stratify=remaining_labels, random_state=42)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=8, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    
    # train_dataset, val_test_dataset = train_test_split(dataset, test_size=test_ratio, random_state=42)
    # validation_dataset, test_dataset = train_test_split(val_test_dataset, test_size=0.5, random_state=42)
    # # Create data loaders for train and test sets
    # train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
    # validation_dataloader = DataLoader(validation_dataset, batch_size=8, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    dataloaders = {'train': train_dataloader, 'val': validation_dataloader, 'test': test_dataloader}
    return dataloaders
