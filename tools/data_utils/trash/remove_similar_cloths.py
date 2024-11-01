import os
from PIL import Image
import imagehash

def get_image_size(image_path):
    return os.path.getsize(image_path)

def sort_images_by_size(folder_path):
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    images.sort(key=lambda x: get_image_size(os.path.join(folder_path, x)))
    return images

def are_images_similar(image1_path, image2_path):
    hash1 = imagehash.average_hash(Image.open(image1_path))
    hash2 = imagehash.average_hash(Image.open(image2_path))
    return hash1 - hash2 < 1  # Allow a hash difference threshold

def delete_similar_images(folder_path):
    images = sort_images_by_size(folder_path)

    if not images:
        print("No images found in the folder.")
        return

    previous_image_path = os.path.join(folder_path, images[0])
    previous_size = get_image_size(previous_image_path)

    for i in range(1, len(images)):
        current_image_path = os.path.join(folder_path, images[i])
        current_size = get_image_size(current_image_path)

        size_difference = abs(current_size - previous_size) / previous_size * 10000

        if size_difference < 1 and are_images_similar(previous_image_path, current_image_path):
            print(f"Deleting {images[i]} due to size similarity or visual similarity.")
            os.remove(current_image_path)
        else:
            previous_image_path = current_image_path
            previous_size = current_size

folder_path = '/home/jupyter/data/human_detection/screenshots_jpg'
delete_similar_images(folder_path)


# folder_path = '/home/jupyter/data/human_detection/screenshots_jpg'
# delete_similar_images(folder_path)
