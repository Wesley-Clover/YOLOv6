import os
import random
import shutil


# Define the path to the directory containing the text files
# labels_path1 = "/home/jupyter/data/crowdhuman/CrowdHuman_train01_c/labels"
use_random =False

# Function to adjust values according to the specified rules
def adjust_values(value):
    value = float(value)
    if value < 0:
        return 0.0
    elif value >= 1:
        return 0.99
    return value

def adjust_value0(value):

    return 0

if False:
    temo_file = os.listdir(labels_path1)
    # Iterate over all .txt files in the folder
    for filename in os.listdir(labels_path1):
        if filename.endswith('.txt'):
            file_path = os.path.join(labels_path1, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Process each line in the file
            adjusted_lines = []
            for line in lines:
                # Split line into individual values, adjust each, and rejoin
                values = line.split()
                adjusted_values = ['0'] + [str(adjust_values(value)) for value in values[1:]]
                adjusted_lines.append(' '.join(adjusted_values) + '\n')

            # Write the adjusted lines back to the file
            with open(file_path, 'w') as file:
                file.writelines(adjusted_lines)

"Processing complete"

# Define paths
base_path = "/home/jupyter/data/human_detection/raw_datasets/ptracker/pythonProject/dataset/personpath22/processed_data"
images_path = os.path.join(base_path, "data")
labels_path = os.path.join(base_path, "anns")

base_path = "/home/jupyter/data/human_detection/raw_datasets/ptracker/pythonProject/dataset/personpath22/processed_data/data_6"

# Define new paths for train and val splits
train_images_path = os.path.join(base_path, "images", "train")
val_images_path = os.path.join(base_path, "images", "val")
train_labels_path = os.path.join(base_path, "labels", "train")
val_labels_path = os.path.join(base_path, "labels", "val")

# Create directories for train and val sets
os.makedirs(train_images_path, exist_ok=True)
os.makedirs(val_images_path, exist_ok=True)
os.makedirs(train_labels_path, exist_ok=True)
os.makedirs(val_labels_path, exist_ok=True)

# List all images and labels
image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png'))]
label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]

# Ensure matching images and labels
image_files = sorted(image_files)
label_files = sorted(label_files)

if use_random:
    # Randomly select 900 images for the validation set
    val_images = random.sample(image_files, int(len(image_files)*0.13))

    # Train images are the rest
    train_images = [img for img in image_files if img not in val_images]

else:
    # Define the number of validation samples
    num_val_samples = int(len(image_files)*0.13)

    # Ensure there are enough images
    if len(image_files) < num_val_samples:
        raise ValueError(f"Not enough images to create a validation set of {num_val_samples} samples.")

    # Select the last 2,000 images for validation
    val_images = image_files[-num_val_samples:]

    # Assign the remaining images to the training set
    train_images = image_files[:-num_val_samples]

# Copy files to the respective folders
for image in train_images:
    shutil.copy(os.path.join(images_path, image), os.path.join(train_images_path, image))
    label = image.replace('.jpg', '.txt').replace('.png', '.txt')
    if label in label_files:
        shutil.copy(os.path.join(labels_path, label), os.path.join(train_labels_path, label))

for image in val_images:
    shutil.copy(os.path.join(images_path, image), os.path.join(val_images_path, image))
    label = image.replace('.jpg', '.txt').replace('.png', '.txt')
    if label in label_files:
        shutil.copy(os.path.join(labels_path, label), os.path.join(val_labels_path, label))

# Display the number of files moved to verify
train_image_count = len(os.listdir(train_images_path))
val_image_count = len(os.listdir(val_images_path))

train_label_count = len(os.listdir(train_labels_path))
val_label_count = len(os.listdir(val_labels_path))

(train_image_count, val_image_count, train_label_count, val_label_count)

