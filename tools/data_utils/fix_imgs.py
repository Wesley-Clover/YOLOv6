import cv2
import os, subprocess

def check_and_delete_images(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith('.png') or filename.lower().endswith('.jpg'):
            file_path = os.path.join(directory, filename)
            # Run the process_image.py script and capture the output
            result = subprocess.run(['python', 'process_image.py', file_path], capture_output=True, text=True)
            # Check if the specific libpng warning was output
            if "iCCP: known incorrect sRGB profile" in result.stderr:
                # If the warning is present, delete the image
                os.remove(file_path)
                print(f"Deleted {filename} due to iCCP warning")
            elif result.stdout:
                # Print any other output from the script
                print(result.stdout)
            else:
                print(f"Kept {filename}")

# Specify the directory containing your PNG images
directory = '/home/jupyter/data/crowdhuman/cloths_2020/images/train'
check_and_delete_images(directory)
