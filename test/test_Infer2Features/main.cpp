#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <memory>
#include <NvOnnxParser.h>
#include <vector>
#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
// #include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
// #include <opencv2/cudaarithm.hpp>
#include <algorithm>
#include <numeric>

const char* INPUT_BLOB_NAME = "images";
const char* OUTPUT_BLOB_NAME = "features";

// static const int INPUT_SIZE = ;
static const int OUTPUT_SIZE = 128;


#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

// utilities ----------------------------------------------------------------------------------------------------------
// class to log errors, warnings, and other information during the build and inference phases
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) override {
        // remove this 'if' if you need more logged info
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
            std::cout << msg << "\n";
        }
    }
} gLogger;

// destroy TensorRT objects if something goes wrong
struct TRTDestroy
{
    template <class T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

template <class T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;
// Tạo con trỏ độc quyền để quản lý một đối tượng (cụ thể là các đối tượng TRT)

// calculate size of tensor
size_t getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

// get classes names
std::vector<std::string> getClassNames(const std::string& imagenet_classes)
{
    std::ifstream classes_file(imagenet_classes);
    std::vector<std::string> classes;
    if (!classes_file.good())
    {
        std::cerr << "ERROR: can't read file with classes names.\n";
        return classes;
    }
    std::string class_name;
    while (std::getline(classes_file, class_name))
    {
        classes.push_back(class_name);
    }
    return classes;
}

// preprocessing stage ------------------------------------------------------------------------------------------------
// void preprocessImage(const std::string& image_path, float* gpu_input, const nvinfer1::Dims& dims)
// {
//     // read input image
//     cv::Mat frame = cv::imread(image_path);
//     if (frame.empty())
//     {
//         std::cerr << "Input image " << image_path << " load failed\n";
//         return;
//     }
//     cv::cuda::GpuMat gpu_frame;
//     // upload image to GPU
//     gpu_frame.upload(frame);

//     auto input_width = dims.d[2];
//     auto input_height = dims.d[1];
//     auto channels = dims.d[0];
//     auto input_size = cv::Size(input_width, input_height);
//     // resize
//     cv::cuda::GpuMat resized;
//     cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_NEAREST);
//     // normalize
//     cv::cuda::GpuMat flt_image;
//     resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
//     cv::cuda::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
//     cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);
//     // to tensor
//     std::vector<cv::cuda::GpuMat> chw;
//     for (size_t i = 0; i < channels; ++i)
//     {
//         chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + i * input_width * input_height));
//     }
//     cv::cuda::split(flt_image, chw);
// }

// post-processing stage ----------------------------------------------------------------------------------------------
void postprocessResults(float *gpu_output, const nvinfer1::Dims &dims, int batch_size)
{
    // get class names
    auto classes = getClassNames("imagenet_classes.txt");

    // copy results from GPU to CPU
    std::vector<float> cpu_output(getSizeByDim(dims) * batch_size);
    cudaMemcpy(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // calculate softmax
    std::transform(cpu_output.begin(), cpu_output.end(), cpu_output.begin(), [](float val) {return std::exp(val);});
    auto sum = std::accumulate(cpu_output.begin(), cpu_output.end(), 0.0);
    // find top classes predicted by the model
    std::vector<int> indices(getSizeByDim(dims) * batch_size);
    std::iota(indices.begin(), indices.end(), 0); // generate sequence 0, 1, 2, 3, ..., 999
    std::sort(indices.begin(), indices.end(), [&cpu_output](int i1, int i2) {return cpu_output[i1] > cpu_output[i2];});
    // print results
    int i = 0;
    while (cpu_output[indices[i]] / sum > 0.005)
    {
        if (classes.size() > indices[i])
        {
            std::cout << "class: " << classes[indices[i]] << " | ";
        }
        std::cout << "confidence: " << 100 * cpu_output[indices[i]] / sum << "% | index: " << indices[i] << "\n";
        ++i;
    }
}

// initialize TensorRT engine and parse ONNX model --------------------------------------------------------------------
void parseOnnxModel(const std::string& model_path, TRTUniquePtr<nvinfer1::ICudaEngine>& engine,
                    TRTUniquePtr<nvinfer1::IExecutionContext>& context)
{
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TRTUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(gLogger)};
    TRTUniquePtr<nvinfer1::INetworkDefinition> network {builder->createNetworkV2(explicitBatch)};
    // TRTUniquePtr<nvinfer1::INetworkDefinition> network{builder->createNetwork()};
    TRTUniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, gLogger)};
    TRTUniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    // parse ONNX
    if (!parser->parseFromFile(model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
    {
        std::cerr << "ERROR: could not parse the model.\n";
        return;
    }
    else
    {
        std::cout << "SUCCESSFUL: parsering the model is successful.\n";
    }
    // allow TensorRT to use up to 1GB of GPU memory for tactic selection.
    config->setMaxWorkspaceSize(1ULL << 30);
    // use FP16 mode if possible
    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    // we have only one image in batch
    builder->setMaxBatchSize(1);
    // generate TensorRT engine optimized for the target platform
    engine.reset(builder->buildEngineWithConfig(*network, *config));
    context.reset(engine->createExecutionContext());
}

void toBuffer(const std::vector<cv::Mat> &mats_vec, uchar *buf) 
{
	int pos = 0;
	for(const cv::Mat& img : mats_vec) 
	{
		int Lenth = img.rows * img.cols * 3;
		int nr = img.rows;
		int nc = img.cols;
		if(img.isContinuous()) 
		{
			nr = 1;
			nc = Lenth;
		}
		for(int i = 0; i < nr; i++) 
		{
			const uchar* inData = img.ptr<uchar>(i);
			for(int j = 0; j < nc; j++) 
			{
				buf[pos] = *inData++;
				pos++;
			}
		}//end for
	}//end imgs;
};

// void doInference(TRTUniquePtr<nvinfer1::IExecutionContext>& context, TRTUniquePtr<nvinfer1::ICudaEngine>& engine,
//                   std::vector<cv::Mat>& mats_vec, float* output) 
// {
//     // Pointers to input and output device buffers to pass to engine.
//     // Engine requires exactly IEngine::getNbBindings() number of buffers.
//     assert(engine->getNbBindings() == 2);
//     void* buffers[2];

//     // In order to bind the buffers, we need to know the names of the input and output tensors.
//     // Note that indices are guaranteed to be less than IEngine::getNbBindings()
//     const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
//     const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);

//     // Create GPU buffers on device
//     int COUNT = mats_vec.size();
//     const int WIDTH = 64;
//     const int HEIGHT = 128;
//     const int DEEP = 3;
//     const int FEATURES = 64;
//     CHECK(cudaMalloc(&buffers[inputIndex], COUNT * WIDTH * HEIGHT * DEEP * sizeof(uchar)));
//     CHECK(cudaMalloc(&buffers[outputIndex], COUNT * FEATURES * sizeof(float)));

//     // Create stream
//     cudaStream_t stream;
//     CHECK(cudaStreamCreate(&stream));

//     // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
//     uchar* input_buff;
//     toBuffer(mats_vec, input_buff);
//     CHECK(cudaMemcpyAsync(buffers[inputIndex], input_buff, COUNT * WIDTH * HEIGHT * DEEP * sizeof(uchar), cudaMemcpyHostToDevice, stream));
//     context->enqueue(1, buffers, stream, nullptr);
//     CHECK(cudaMemcpyAsync(output, buffers[outputIndex], COUNT * FEATURES * sizeof(float), cudaMemcpyDeviceToHost, stream));
//     cudaStreamSynchronize(stream);

//     // Release stream and buffers
//     cudaStreamDestroy(stream);
//     CHECK(cudaFree(buffers[inputIndex]));
//     CHECK(cudaFree(buffers[outputIndex]));
// }

// main pipeline ------------------------------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // if (argc < 3)
    // {
    //    std::cerr << "usage: " << argv[0] << " model.onnx image.jpg\n";
    //    return -1;
    // }
    std::string model_path(argv[1]); 
    // path model *.onnx format: "/home/thaivu/Projects/Turnstiles_Fare_Evasion/RUNNINGDATA/ONNX_networks/mars-small128.onnx"
    // std::string image_path(argv[2]); 
    // path a test image: "/home/thaivu/Projects/TestImages/test1_7human.jpg"
    int batch_size = 1;

    // initialize TensorRT engine and parse ONNX model
    TRTUniquePtr<nvinfer1::ICudaEngine> engine{nullptr};
    TRTUniquePtr<nvinfer1::IExecutionContext> context{nullptr};
    parseOnnxModel(model_path, engine, context);
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    // get sizes of input and output and allocate memory required for input data and for output data
    std::vector<nvinfer1::Dims> input_dims; // we expect only one input
    std::vector<nvinfer1::Dims> output_dims; // and one output
    // std::vector<void*> buffers(engine->getNbBindings()); // buffers for input and output data
    // do ở đây chỉ có 1 input và 1 output nên engine->getNbBindings() = 2
    // for (size_t i = 0; i < engine->getNbBindings(); ++i)
    // {
    //     auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * batch_size * sizeof(float);
    //     cudaMalloc(&buffers[i], binding_size);
    //     if (engine->bindingIsInput(i))
    //     {
    //         input_dims.emplace_back(engine->getBindingDimensions(i));
    //     }
    //     else
    //     {
    //         output_dims.emplace_back(engine->getBindingDimensions(i));
    //     }
    // }
    std::cout << "inputIndex, outputIndex: " << inputIndex << ", " << outputIndex << std::endl;
    input_dims.emplace_back(engine->getBindingDimensions(inputIndex));
    output_dims.emplace_back(engine->getBindingDimensions(outputIndex));
    if (input_dims.empty() || output_dims.empty())
    {
        std::cerr << "Expect at least one input and one output for network\n";
        return -1;
    }
    else
    {
        std::cout << "Final!!! OK" << std::endl;
    }

    // đầu vào là 1 vector chứa 1 tập các đối tượng cv::Mat
    // trong main flow sẽ phải trích xuất các đối tượng cv::Mat ra ứng với từng detection rồi nhét vào vector này
    // doInference()

    // // preprocess input data
    // preprocessImage(image_path, (uint8_t *) buffers[0], input_dims[0]);
    // // inference
    // context->enqueue(batch_size, buffers.data(), 0, nullptr);
    // // postprocess results
    // postprocessResults((float *) buffers[1], output_dims[0], batch_size);

    // for (void* buf : buffers)
    // {
    //    cudaFree(buf);
    // }
    return 0;
}