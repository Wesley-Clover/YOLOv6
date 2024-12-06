#pragma once

#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class YoloV6TensorRT
{
public:
    struct Object
    {
        cv::Rect_<float> boundingBox;
        int classId;
        float probability;
    };

    YoloV6TensorRT(const std::string &enginePath);
    ~YoloV6TensorRT();

    std::vector<Object> detect(const cv::Mat &image);
    void drawAndShowObjects(cv::Mat &bgr, const std::vector<Object> &objects, bool displayOutput);

private:
    static constexpr int DEVICE = 0;
    static constexpr float NMS_THRESH = 0.45f;
    static constexpr float BBOX_CONF_THRESH = 0.5f;
    static constexpr int INPUT_WIDTH = 832;
    static constexpr int INPUT_HEIGHT = 480;
    static constexpr int NUM_CLASSES = 1;

    nvinfer1::IRuntime *runtime{nullptr};
    nvinfer1::ICudaEngine *engine{nullptr};
    nvinfer1::IExecutionContext *context{nullptr};
    float *probabilities{nullptr};
    int outputSize{0};

    cv::Mat staticResize(cv::Mat &img);
    float *createBlobFromImage(cv::Mat &img);
    void performInference(float *input, float *output, cv::Size inputShape);
    void decodeOutputs(float *probabilities, std::vector<Object> &objects, float scale, int imgWidth, int imgHeight);

    static void generateYoloProposals(float *featureMap, int outputSize, float probThreshold,
                                      std::vector<Object> &objects);
    void quickSortDescending(std::vector<Object> &objects, int left, int right);
    static void nmsSortedBboxes(const std::vector<Object> &objects, std::vector<int> &picked, float nmsThreshold);
};