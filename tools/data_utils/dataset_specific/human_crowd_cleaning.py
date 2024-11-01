import os
import shutil

# get wrongly annotated ones from this address: https://appsignal-training-data.s3.us-east-1.amazonaws.com/prepared/human_crowd/humancrowd_wrong.zip
# cleaned humancrowd is the source directory
source_directory = '/home/jupyter/data/crowdhuman/humancrowd_wrong/all'
destination_directory = '/home/jupyter/data/crowdhuman/CrowdHuman_train01/yolov6_data/data_6/images/train'
label_directory = '/home/jupyter/data/crowdhuman/CrowdHuman_train01/yolov6_data/data_6/labels/train'


image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
text_extension = '.txt'

# Store all image filenames (without extensions) from the source directory
source_images = {os.path.splitext(file)[0].lower().split('_')[1] for file in os.listdir(source_directory) if file.lower().endswith(image_extensions)}

# Check images in the destination directory
for file in os.listdir(destination_directory):
    if file.lower().endswith(image_extensions):
        if os.path.splitext(file)[0].lower() in source_images:
            os.remove(os.path.join(destination_directory, file))
            print(f'Deleted image: {file} from destination directory.')

# Check text files in the label directory
for file in os.listdir(label_directory):
    if file.lower().endswith(text_extension):
        if os.path.splitext(file)[0].lower() in source_images:
            os.remove(os.path.join(label_directory, file))
            print(f'Deleted text file: {file} from label directory.')

print("Cleanup completed.")
