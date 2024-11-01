import os
import shutil

# List of paths to the 7 datasets
dataset_paths = [
    # '/home/jupyter/data/human_detection/raw_datasets/JAD/data',
    '/home/jupyter/data/crowdhuman/CrowdHuman_train01/temp',
    '/home/jupyter/data/human_detection/raw_datasets/MOT/MOT17/data',
    '/home/jupyter/data/human_detection/raw_datasets/MOT/MOT20/data'
    # '/path/to/dataset5',
    # '/path/to/dataset6',
    # '/path/to/dataset7'
]

# Define the number of frames to skip for each dataset path
skip_frames = [0, 0, 0, 0, 0, 0, 0]  # Example skipping patterns

# Destination directories
destination_images = '/home/jupyter/data/crowdhuman/CrowdHuman_train01_c/images'
destination_labels = '/home/jupyter/data/crowdhuman/CrowdHuman_train01_c/labels'

# Create destination directories if they do not exist
os.makedirs(destination_images, exist_ok=True)
os.makedirs(destination_labels, exist_ok=True)

def copy_files(source_dir, dest_dir, skip_count):
    files = sorted(os.listdir(source_dir))
    for i, file in enumerate(files):
        if i%100==0:
            print(i)
        if i % (skip_count + 1) == 0:  # Skip the desired number of frames
            source_file = os.path.join(source_dir, file)
            dest_file = os.path.join(dest_dir, file)
            shutil.copy(source_file, dest_file)

# Loop through all datasets
for idx, dataset in enumerate(dataset_paths):
    print(dataset)
    images_path = os.path.join(dataset, 'images')
    labels_path = os.path.join(dataset, 'labels')

    # Get the skip count for the current dataset
    skip_count = skip_frames[idx]

    # Copy images and labels with frame skipping
    copy_files(images_path, destination_images, skip_count)
    copy_files(labels_path, destination_labels, skip_count)

print("Files copied successfully.")
