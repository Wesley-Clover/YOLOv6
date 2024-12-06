#include <chrono>
#include <iostream>
#include "yolov6_class_edgesignal.hpp"

// Function to calculate and print average FPS
void fpsCalculator(int frameCount, std::chrono::steady_clock::time_point &startTime)
{
    if (frameCount % 100 == 0)
    {
        auto endTime = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsedSeconds = endTime - startTime;
        double fps = 100.0 / elapsedSeconds.count();
        std::cout << "Average FPS over last 100 frames: " << fps << std::endl;
        startTime = endTime; // Reset startTime for the next 100 frames
    }
}

int main(int argc, char **argv)
{
    if (argc < 4 || (std::string(argv[2]) != "-i" && std::string(argv[2]) != "-v"))
    {
        std::cerr << "Usage: " << argv[0] << " <engine_file> -i <input_image> | -v <video_path> "
                  << "[--show-processing-fps] [--show-average-fps] [--no-display] [--save-video <output_path>]" << std::endl;
        return -1;
    }

    // Add flags for FPS calculations and display control
    bool showProcessingFps = false;
    bool showAverageFps = false;
    bool displayOutput = true; // Default to showing output
    std::string outputVideoPath = "";

    // Parse optional arguments
    for (int i = 4; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--show-processing-fps")
            showProcessingFps = true;
        else if (arg == "--show-average-fps")
            showAverageFps = true;
        else if (arg == "--no-display")
            displayOutput = false;
        else if (arg == "--save-video" && i + 1 < argc)
        {
            outputVideoPath = argv[i + 1];
            i++; // Skip next argument as it's the output path
        }
    }

    try
    {
        // Initialize detector
        YoloV6TensorRT detector(argv[1]);

        if (std::string(argv[2]) == "-i")
        {
            // Load and process image
            cv::Mat img = cv::imread(argv[3]);
            if (img.empty())
            {
                throw std::runtime_error("Failed to load image: " + std::string(argv[3]));
            }

            // Perform detection
            auto objects = detector.detect(img);

            // Draw results
            detector.drawAndShowObjects(img, objects, displayOutput);
            cv::imwrite("./result.jpg", img);

            std::cout << "Image detection completed successfully" << std::endl;
        }
        else if (std::string(argv[2]) == "-v")
        {
            // Open video file
            cv::VideoCapture cap(argv[3]);
            if (!cap.isOpened())
            {
                throw std::runtime_error("Failed to open video: " + std::string(argv[3]));
            }

            // Get video properties
            int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
            int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
            double fps = cap.get(cv::CAP_PROP_FPS);

            // Initialize video writer if output path is specified
            cv::VideoWriter videoWriter;
            if (!outputVideoPath.empty())
            {
                videoWriter.open(outputVideoPath,
                                 cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                 fps,
                                 cv::Size(frame_width, frame_height));

                if (!videoWriter.isOpened())
                {
                    throw std::runtime_error("Failed to create output video file: " + outputVideoPath);
                }
            }

            cv::Mat frame;
            int frameCount = 0;
            auto startTime = std::chrono::steady_clock::now();

            while (cap.read(frame))
            {
                auto startTimeProcessingFps = std::chrono::steady_clock::now();
                auto objects = detector.detect(frame);
                detector.drawAndShowObjects(frame, objects, displayOutput);

                // Write frame to output video if enabled
                if (videoWriter.isOpened())
                {
                    videoWriter.write(frame);
                }

                if (showProcessingFps)
                {
                    auto endTimeProcessingFps = std::chrono::steady_clock::now();
                    std::chrono::duration<double> elapsedSecondsForProcessingFps =
                        endTimeProcessingFps - startTimeProcessingFps;
                    double processingFps = 1.0 / elapsedSecondsForProcessingFps.count();
                    std::cout << "Processing FPS: " << processingFps << std::endl;
                }

                frameCount++;
                if (showAverageFps)
                {
                    fpsCalculator(frameCount, startTime);
                }
            }

            // Release video writer
            if (videoWriter.isOpened())
            {
                videoWriter.release();
            }

            std::cout << "Video detection completed successfully" << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}