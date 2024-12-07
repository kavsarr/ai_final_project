import os
import shutil
import random

# Path to the dataset
dataset_dir = 'datasets/handwritten-english-characters-and-digits/combined_folder'

# Paths for the split dataset
train_dir = 'datasets/demo_dataset/train'
val_dir = 'datasets/demo_dataset/val'
test_dir = 'datasets/demo_dataset/test'

# Create the split directories
for split_dir in [train_dir, val_dir, test_dir]:
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
        
all_data = {}

# Loop over classes
for split in os.listdir(dataset_dir):
    for class_ in os.listdir(os.path.join(dataset_dir, split)):
        class_path = os.path.join(dataset_dir, split, class_)
        
        # Check if it's a directory (class folder)
        if os.path.isdir(class_path):
            # Get a list of all images in the species folder
            images = os.listdir(class_path)
            images = [os.path.join(class_path, i) for i in images]
            
            try:
                all_data[class_] += images
            except:
                all_data[class_] = images
            
            
for class_ in all_data:
    # Create the corresponding directories for train, val, test
    species_train_dir = os.path.join(train_dir, class_)
    species_val_dir = os.path.join(val_dir, class_)
    species_test_dir = os.path.join(test_dir, class_)
    
    # Create class folders inside each split folder
    for split_folder in [species_train_dir, species_val_dir, species_test_dir]:
        if not os.path.exists(split_folder):
            os.makedirs(split_folder)
            
    images = all_data[class_]
    
    # Split the images into 80%, 10%, and 10%
    total_images = len(images)
    train_size = int(total_images * 0.8)
    val_size = int(total_images * 0.1)
    
    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]
    
    # Move images to the respective directories
    for src in train_images:
        img = src.split('/')[-1]
        dst = os.path.join(species_train_dir, img)
        shutil.copy(src, dst)
        
    for src in val_images:
        img = src.split('/')[-1]
        dst = os.path.join(species_val_dir, img)
        shutil.copy(src, dst)
        
    for src in test_images:
        img = src.split('/')[-1]
        dst = os.path.join(species_test_dir, img)
        shutil.copy(src, dst)

print("Data split completed!")
