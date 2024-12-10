# YOLOv6-TensorRT in C++

## Dependencies
- TensorRT-8.2.3.0 or other versions 8.x
- OpenCV-4.1.0 or later

## Install nvidia container toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

## Build TensorRT container
docker build -f Dockerfile_tensorrt_x86 -t tensorrt_dev .

## Run TensorRT container
docker run --gpus all -it tensorrt_dev

## Copy trained yolov6 model to container
docker cp /path/to/yolov6/pt/model CONTAINER_ID:/workspace/Yolov6/deploy/TensorRT/model

## Convert PT model to ONNX
```bash
chmod +x deploy/TensorRT/convert_pt2_onnx.sh
```

Navigate to the TensorRT directory and use the conversion script:
```bash
cd deploy/TensorRT
./convert_pt2_onnx.sh --weights model/your_model.pt
```

The script supports various parameters:
```bash
./convert_pt2_onnx.sh --help  # Show all available options

# Example with common parameters:
./convert_pt2_onnx.sh \
    --weights model/your_model.pt \
    --img-size 480 832 \
    --device 0
```

Key Parameters:
- `--weights`: Path to your PT model file
- `--img-size`: Input image size (default: 480, 832)
- `--batch-size`: Batch size for inference (default: 1)
- `--device`: GPU device ID (default: 0)
- `--end2end`: Enable end2end ONNX export
- `--simplify`: Simplify ONNX model
- `--half`: Enable FP16 precision
- `--dynamic-batch`: Enable dynamic batch size

After conversion, you'll find the ONNX model in the same directory as your PT model with the .onnx extension.

## Convert ONNX to TensorRT
After obtaining the ONNX model, you can convert it to TensorRT format. First, make the conversion script executable:

```bash
chmod +x deploy/TensorRT/convert_onnx2_trt.sh
```

Navigate to the TensorRT directory and use the conversion script:
```bash
cd deploy/TensorRT
./convert_onnx2_trt.sh --model model/your_model.onnx --dtype fp32
```

The script supports various parameters:
```bash
./convert_onnx2_trt.sh --help  # Show all available options

# Example with FP32 precision:
./convert_onnx2_trt.sh \
    --model model/your_model.onnx \
    --dtype fp32

# Example with FP16 precision:
./convert_onnx2_trt.sh \
    --model model/your_model.onnx \
    --dtype fp16

# Example with INT8 precision (requires calibration):
./convert_onnx2_trt.sh \
    --model model/your_model.onnx \
    --dtype int8 \
    --img-height 480 \
    --img-width 832 \
    --batch-size 128 \
    --num-calib-batch 6
```

Key Parameters:
- `--model`: Path to your ONNX model file
- `--dtype`: Precision type (fp32, fp16, or int8)
- `--img-height`: Image height for INT8 calibration (default: 480)
- `--img-width`: Image width for INT8 calibration (default: 832)
- `--batch-size`: Batch size for INT8 calibration (default: 128)
- `--num-calib-batch`: Number of calibration batches for INT8 (default: 6)
- `--calib-img-dir`: Directory containing calibration images
- `--verbose`: Enable verbose output for debugging
- `--qat`: Enable Quantization Aware Training mode

After conversion, you'll find the TensorRT engine file (.trt) in the same directory as your ONNX model. For INT8 models, the filename will include calibration details.

Note: INT8 conversion requires a calibration dataset and additional parameters. Make sure you have the appropriate calibration images available when using INT8 precision.


## Build TensorRT C++ code
```bash
cd deploy/TensorRT/yolov6_edgesignal
mkdir build
cd build
cmake ..
make
```
## Alternative: Build TensorRT C++ code with script
First, make the build script executable:
```bash
chmod +x deploy/TensorRT/build_cpp_files.sh
```

Then run the build script:
```bash
cd deploy/TensorRT
./build.sh
```

The script supports various options:
```bash
./build.sh --help  # Show all available options

# Example with custom build directory:
./build.sh --build-dir custom_build

# Example with Debug build type:
./build.sh --build-type Debug

# Example with clean build:
./build.sh --clean
```

Key Parameters:
- `--build-dir`: Specify custom build directory name (default: build)
- `--build-type`: Set build type to Debug or Release (default: Release)
- `--clean`: Remove existing build directory before building
- `--help`: Show help message

After successful build, the executable will be available in the build directory.

## Run TensorRT C++ Application
After building the application, you can run it with the following syntax:

```bash
./yolov6 <engine_file> -i <input_image> | -v <video_path> [options]
```

Examples:

1. Process a single image:
```bash
# Display result
./yolov6 model/best_stop_aug_ckpt.trt -i test_images/test.jpg

# Process without displaying
./yolov6 model/best_stop_aug_ckpt.trt -i test_images/test.jpg --no-display
```

2. Process a video:
```bash
# Display result
./yolov6 model/best_stop_aug_ckpt.trt -v test_videos/test.mp4

# Save processed video
./yolov6 model/best_stop_aug_ckpt.trt -v test_videos/test.mp4 --save-video output.avi

# Process without displaying
./yolov6 model/best_stop_aug_ckpt.trt -v test_videos/test.mp4 --no-display

# Process with all options
./yolov6 model/best_stop_aug_ckpt.trt -v test_videos/test.mp4 \
    --show-processing-fps \
    --show-average-fps \
    --save-video output.avi
```

Parameters:
- `<engine_file>`: Path to your TensorRT engine file (.trt)
- `-i <input_image>`: Process a single image
- `-v <video_path>`: Process a video file
- `--show-processing-fps`: Show FPS for each frame processing (optional)
- `--show-average-fps`: Show average FPS over every 100 frames (optional)
- `--no-display`: Disable OpenCV window display (optional)
- `--save-video <output_path>`: Save processed video to specified path (optional)

Note: 
- Press any key to exit when processing video
- For image processing, the window will stay open until a key is pressed
- FPS monitoring options are particularly useful for performance testing
- Video output is saved in MJPG format
