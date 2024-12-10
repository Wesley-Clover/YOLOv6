#include "yolov6_class_edgesignal.hpp"
#include <cuda_runtime_api.h>
#include <fstream>
#include "logging.h"

#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

static Logger gLogger;

YoloV6TensorRT::YoloV6TensorRT(const std::string &enginePath)
{
    cudaSetDevice(DEVICE);

    // Load engine file
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good())
    {
        throw std::runtime_error("Failed to load engine file: " + enginePath);
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);

    runtime = nvinfer1::createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(engineData.data(), size);
    context = engine->createExecutionContext();

    auto outDims = engine->getBindingDimensions(1);
    outputSize = 1;
    for (int j = 0; j < outDims.nbDims; j++)
    {
        outputSize *= outDims.d[j];
    }
    probabilities = new float[outputSize];
}

YoloV6TensorRT::~YoloV6TensorRT()
{
    delete[] probabilities;
    if (context)
        context->destroy();
    if (engine)
        engine->destroy();
    if (runtime)
        runtime->destroy();
}

cv::Mat YoloV6TensorRT::staticResize(cv::Mat &img)
{
    float r = std::min(INPUT_WIDTH / (img.cols * 1.0), INPUT_HEIGHT / (img.rows * 1.0));
    int unpadWidth = r * img.cols;
    int unpadHeight = r * img.rows;
    cv::Mat re(unpadHeight, unpadWidth, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

float *YoloV6TensorRT::createBlobFromImage(cv::Mat &img)
{
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    float *blob = new float[img.total() * 3];
    int channels = 3;
    int imgH = img.rows;
    int imgW = img.cols;

    for (size_t c = 0; c < channels; c++)
    {
        for (size_t h = 0; h < imgH; h++)
        {
            for (size_t w = 0; w < imgW; w++)
            {
                blob[c * imgW * imgH + h * imgW + w] = (((float)img.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
    return blob;
}

void YoloV6TensorRT::performInference(float *input, float *output, cv::Size size)
{
    int inputIndex = engine->getBindingIndex("images");
    int outputIndex = engine->getBindingIndex("outputs");

    // Prepare buffers
    void *buffers[2];
    CHECK(cudaMalloc(&buffers[inputIndex], 1 * 3 * size.height * size.width * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], 1 * 8190 * 6 * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Set input dimensions
    nvinfer1::Dims inputDims = engine->getBindingDimensions(inputIndex);
    inputDims.d[2] = size.height;
    inputDims.d[3] = size.width;
    context->setBindingDimensions(inputIndex, inputDims);

    // Copy input to device
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 1 * 3 * size.height * size.width * sizeof(float),
                          cudaMemcpyHostToDevice, stream));

    // Enqueue inference
    context->enqueueV2(buffers, stream, nullptr);

    // Copy output back to host
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], 1 * 8190 * 6 * sizeof(float), cudaMemcpyDeviceToHost, stream));

    // Synchronize and clean up
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

void YoloV6TensorRT::generateYoloProposals(float *featureMap, int outputSize, float probThreshold,
                                           std::vector<Object> &objects)
{
    auto detections = outputSize / (NUM_CLASSES + 5);
    for (int boxsIdx = 0; boxsIdx < detections; boxsIdx++)
    {
        const int basicPos = boxsIdx * (NUM_CLASSES + 5);
        float xCenter = featureMap[basicPos + 0];
        float yCenter = featureMap[basicPos + 1];
        float w = featureMap[basicPos + 2];
        float h = featureMap[basicPos + 3];
        float x0 = xCenter - w * 0.5f;
        float y0 = yCenter - h * 0.5f;
        float boxObjectness = featureMap[basicPos + 4];

        float boxClsScore = featureMap[basicPos + 5];
        float boxProb = boxObjectness * boxClsScore;

        if (boxProb > probThreshold)
        {
            Object obj;
            obj.boundingBox = cv::Rect_<float>(x0, y0, w, h);
            obj.classId = 0;
            obj.probability = boxProb;
            objects.push_back(obj);
        }
    }
}

void YoloV6TensorRT::quickSortDescending(std::vector<Object> &objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].probability;

    while (i <= j)
    {
        while (objects[i].probability > p)
            i++;
        while (objects[j].probability < p)
            j--;

        if (i <= j)
        {
            std::swap(objects[i], objects[j]);
            i++;
            j--;
        }
    }

    if (left < j)
        quickSortDescending(objects, left, j);
    if (i < right)
        quickSortDescending(objects, i, right);
}

void YoloV6TensorRT::nmsSortedBboxes(const std::vector<Object> &objects, std::vector<int> &picked, float nmsThreshold)
{
    picked.clear();
    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].boundingBox.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object &a = objects[i];
        int keep = 1;

        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object &b = objects[picked[j]];

            cv::Rect_<float> intersection = a.boundingBox & b.boundingBox;
            float intersectionArea = intersection.area();
            float unionArea = areas[i] + areas[picked[j]] - intersectionArea;

            if (intersectionArea / unionArea > nmsThreshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

void YoloV6TensorRT::decodeOutputs(float *probabilities, std::vector<Object> &objects, float scale, int imgWidth,
                                   int imgHeight)
{
    std::vector<Object> proposals;
    generateYoloProposals(probabilities, outputSize, BBOX_CONF_THRESH, proposals);

    if (!proposals.empty())
    {
        quickSortDescending(proposals, 0, proposals.size() - 1);

        std::vector<int> picked;
        nmsSortedBboxes(proposals, picked, NMS_THRESH);

        int count = picked.size();
        objects.resize(count);

        for (int i = 0; i < count; i++)
        {
            objects[i] = proposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (objects[i].boundingBox.x) / scale;
            float y0 = (objects[i].boundingBox.y) / scale;
            float x1 = (objects[i].boundingBox.x + objects[i].boundingBox.width) / scale;
            float y1 = (objects[i].boundingBox.y + objects[i].boundingBox.height) / scale;

            // clip
            x0 = std::max(std::min(x0, (float)(imgWidth - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(imgHeight - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(imgWidth - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(imgHeight - 1)), 0.f);

            objects[i].boundingBox = cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0);
        }
    }
}

std::vector<YoloV6TensorRT::Object> YoloV6TensorRT::detect(const cv::Mat &image)
{
    cv::Mat img = image.clone();
    int imgW = img.cols;
    int imgH = img.rows;

    // Preprocess
    cv::Mat prImg = staticResize(img);
    float *blob = createBlobFromImage(prImg);
    float scale = std::min(INPUT_WIDTH / (img.cols * 1.0), INPUT_HEIGHT / (img.rows * 1.0));

    // Inference
    performInference(blob, probabilities, prImg.size());

    // Post-process
    std::vector<Object> objects;
    decodeOutputs(probabilities, objects, scale, imgW, imgH);

    delete[] blob;
    return objects;
}

void YoloV6TensorRT::drawAndShowObjects(cv::Mat &bgr, const std::vector<Object> &objects, bool displayOutput)
{
    static const float colors[1][3] = {{0.000, 1.000, 0.000}};

    for (const auto &obj : objects)
    {
        cv::Scalar color = cv::Scalar(colors[0][0], colors[0][1], colors[0][2]);
        float cMean = cv::mean(color)[0];
        cv::Scalar txtColor = (cMean > 0.5) ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);

        cv::rectangle(bgr, obj.boundingBox, color * 255, 2);

        char text[256];
        sprintf(text, "%.1f%%", obj.probability * 100);

        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);
        cv::Scalar txtBkColor = color * 0.7 * 255;

        int x = obj.boundingBox.x;
        int y = obj.boundingBox.y + 1;

        cv::rectangle(bgr, cv::Rect(cv::Point(x, y), cv::Size(labelSize.width, labelSize.height + baseLine)),
                      txtBkColor, -1);

        cv::putText(bgr, text, cv::Point(x, y + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, txtColor, 1);
    }

    if (displayOutput)
    {
        cv::imshow("Detection Result", bgr);
        cv::waitKey(1);
    }
}
