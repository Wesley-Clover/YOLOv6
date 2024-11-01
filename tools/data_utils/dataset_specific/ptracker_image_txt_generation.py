
import json
import cv2
import os

# Define paths
main_json_path = '/home/jupyter/data/human_detection/raw_datasets/ptracker/pythonProject/dataset/personpath22/annotation/anno_visible_2022.json'  # Adjust the path to your main JSON file
json_directory = '/home/jupyter/data/human_detection/raw_datasets/ptracker/pythonProject/dataset/personpath22/annotation/anno_visible_2022/'  # Adjust to the directory containing individual JSON files

vid_directory = '/home/jupyter/data/human_detection/raw_datasets/ptracker/pythonProject/dataset/personpath22/raw_data/'  # Adjust to the directory containing individual JSON files
dest_image_path = '/home/jupyter/data/human_detection/raw_datasets/ptracker/pythonProject/dataset/personpath22/processed_data/images'  # Adjust the destination directory for images
dest_annotation_path = '/home/jupyter/data/human_detection/raw_datasets/ptracker/pythonProject/dataset/personpath22/processed_data/anns'  # Adjust the destination directory for annotations

# Ensure output directories exist
os.makedirs(dest_image_path, exist_ok=True)
os.makedirs(dest_annotation_path, exist_ok=True)

# Load the main JSON to map videos to their JSON metadata files
with open(main_json_path, 'r') as file:
    main_json = json.load(file)


# Function to convert bounding box coordinates to YOLO format
def convert_to_yolo(width, height, bbox):
    dw = 1. / width
    dh = 1. / height
    x = (bbox[0] + bbox[2] / 2.0) * dw
    y = (bbox[1] + bbox[3] / 2.0) * dh
    w = bbox[2] * dw
    h = bbox[3] * dh
    return [x, y, w, h]


# Process each video based on its metadata JSON file
for video_name, data_path in main_json['samples'].items():
    print(video_name)
    json_file_path = json_directory + video_name + '.json'

    # Load the individual JSON file for the video
    with open(json_file_path, 'r') as file:
        video_data = json.load(file)

    video_path = os.path.join(vid_directory,data_path['metadata']['data_path'])
    width = data_path['metadata']['resolution']['width']
    height = data_path['metadata']['resolution']['height']

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_indices = set(
        entity['blob']['frame_idx'] for entity in video_data['entities'])  # Collect unique frame indices

    for j in sorted(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, j)
        ret, frame = cap.read()
        if not ret:
            continue

        # Save the frame
        frame_filename = f'{video_name[0:-4]}_frame_{j:04d}.jpg'
        frame_path = os.path.join(dest_image_path, frame_filename)
        cv2.imwrite(frame_path, frame)

        # Extract bounding boxes for the current frame
        bounding_boxes = [entity['bb'] for entity in video_data['entities'] if
                          entity['blob']['frame_idx'] == j and entity['labels'].get('reflection') != 1]

        # Convert bounding boxes and save annotations
        annotation_filename = f'{video_name[0:-4]}_frame_{j:04d}.txt'
        annotation_path = os.path.join(dest_annotation_path, annotation_filename)
        with open(annotation_path, 'w') as f:
            for bbox in bounding_boxes:
                yolo_bbox = convert_to_yolo(width, height, bbox)
                # Assuming class ID is always 0, change as needed
                f.write(f"0 {' '.join(map(str, yolo_bbox))}\n")

    cap.release()
