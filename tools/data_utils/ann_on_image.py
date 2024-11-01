import os
import cv2

# Define paths
images_folder = '/home/jupyter/data/human_detection/raw_datasets/ptracker/pythonProject/dataset/personpath22/processed_data/images'  # Path to the folder containing images
labels_folder = '/home/jupyter/data/human_detection/raw_datasets/ptracker/pythonProject/dataset/personpath22/processed_data/anns'  # Path to the folder containing YOLO format labels
output_base_folder = '/home/jupyter/data/human_detection/raw_datasets/ptracker/pythonProject/dataset/personpath22/processed_data/output'  # Path to the destination folder for saving images with annotations


# Function to create batch folders (00, 01, 02...)
def create_batch_folder(batch_idx):
    batch_folder = os.path.join(output_base_folder, f'{batch_idx:02d}')
    if not os.path.exists(batch_folder):
        os.makedirs(batch_folder)
    return batch_folder

# Function to draw bounding boxes
def draw_annotations(image, label_file):
    h, w, _ = image.shape

    with open(label_file, 'r') as file:
        lines = file.readlines()

    for line in lines:
        # YOLO format: class_id, x_center, y_center, width, height (normalized values)
        class_id, x_center, y_center, width, height = map(float, line.strip().split())

        # Convert normalized values to absolute coordinates
        x_center *= w
        y_center *= h
        width *= w
        height *= h

        # Calculate the top-left and bottom-right coordinates
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Draw the bounding box on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image

# Iterate through the images
image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

batch_size = 200
batch_idx = 0
current_batch_folder = create_batch_folder(batch_idx)

for idx, image_file in enumerate(image_files):
    if idx%200==0:
        print(idx)
    image_path = os.path.join(images_folder, image_file)
    label_path = os.path.join(labels_folder, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))

    # Read the image
    image = cv2.imread(image_path)

    # Check if the corresponding label file exists
    if os.path.exists(label_path):
        # Draw annotations on the image
        image = draw_annotations(image, label_path)

    # Determine the folder for the current batch
    if (idx + 1) % batch_size == 1 and idx != 0:
        # Move to the next batch
        batch_idx += 1
        current_batch_folder = create_batch_folder(batch_idx)

    # Save the annotated image in the appropriate batch folder, keeping the original name
    output_path = os.path.join(current_batch_folder, image_file)
    cv2.imwrite(output_path, image)

cv2.destroyAllWindows()

