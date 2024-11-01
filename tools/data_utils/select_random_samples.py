import os
import shutil
import random

def select_and_copy_files(source_dir, dest_dir, sub,num_files=1000):
    # Define supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

    # Define source subdirectories
    source_images_dir = os.path.join(source_dir, 'images',sub)
    source_labels_dir = os.path.join(source_dir, 'labels',sub)

    # Ensure source subdirectories exist
    if not os.path.isdir(source_images_dir):
        raise ValueError(f"Source images directory '{source_images_dir}' does not exist.")
    if not os.path.isdir(source_labels_dir):
        raise ValueError(f"Source labels directory '{source_labels_dir}' does not exist.")

    # Create destination subdirectories if they don't exist
    dest_images_dir = os.path.join(dest_dir, 'images',sub)
    dest_labels_dir = os.path.join(dest_dir, 'labels',sub)
    os.makedirs(dest_images_dir, exist_ok=True)
    os.makedirs(dest_labels_dir, exist_ok=True)

    # List all image files in the source images directory
    all_image_files = [
        f for f in os.listdir(source_images_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]

    # Check if there are enough images
    if len(all_image_files) < num_files:
        print(f"Only {len(all_image_files)} images found. Proceeding with all available images.")
        selected_images = all_image_files
    else:
        selected_images = random.sample(all_image_files, num_files)

    print(f"Selected {len(selected_images)} images.")

    # Copy selected images and their corresponding .txt files
    for image in selected_images:
        image_name, _ = os.path.splitext(image)
        txt_file = image_name + '.txt'

        source_image_path = os.path.join(source_images_dir, image)
        source_txt_path = os.path.join(source_labels_dir, txt_file)

        # Destination paths
        dest_image_path = os.path.join(dest_images_dir, image)
        dest_txt_path = os.path.join(dest_labels_dir, txt_file)

        # Copy image
        shutil.copy2(source_image_path, dest_image_path)

        # Check if corresponding txt file exists
        if os.path.isfile(source_txt_path):
            shutil.copy2(source_txt_path, dest_txt_path)
        else:
            print(f"Warning: Corresponding text file '{txt_file}' not found for image '{image}'.")

    print("Files have been successfully copied.")

# if __name__ == "__main__":
#     # Define your source and destination directories
#     source_dir = '/path/to/source/folder'      # Replace with your source folder path
#     dest_dir = '/path/to/destination/folder'   # Replace with your destination folder path
#
#     # Call the function
#     select_and_copy_files(source_dir, dest_dir, num_files=1000)


if __name__ == "__main__":
    # Define your source and destination directories
    source_dir = '/home/jupyter/data/crowdhuman/data_split'  # Replace with your source folder path
    dest_dir = '/home/jupyter/data/crowdhuman/data_split_sub'  # Replace with your destination folder path

    # Call the function
    select_and_copy_files(source_dir, dest_dir,'val', num_files=1000)
