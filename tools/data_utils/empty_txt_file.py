import os


def create_txt_files_for_images(image_folder,path):
    os.makedirs(path, exist_ok=True)
    # List all files in the image folder
    for filename in os.listdir(image_folder):
        # Check if the file is an image (you can adjust the extensions if needed)
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            # Get the base file name without extension
            base_name = os.path.splitext(filename)[0]
            # Define the path for the new .txt file
            txt_file_path = os.path.join(path, f"{base_name}.txt")

            # Create an empty .txt file
            with open(txt_file_path, 'w') as f:
                pass  # This creates an empty file

            print(f"Created: {txt_file_path}")


# Example usage
image_folder = r'/home/jupyter/data/human_detection/screenshots_jpg'  # Replace with your folder path
dest_path = r'/home/jupyter/data/human_detection/cloths_jpg_txts/anns'
create_txt_files_for_images(image_folder,dest_path)
