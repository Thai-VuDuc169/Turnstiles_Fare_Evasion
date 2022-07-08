#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "common.h"
#include "logging.h"
#include <fstream>
#include <map>
#include <chrono>
#include <cassert>

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

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 128;
static const int INPUT_W = 64;
static const int OUTPUT_SIZE = 128;

#define WEIGHT_MAP_PATH "/home/thaivu169/Projects/Turnstiles_Fare_Evasion/test/gen_wts_deepsort/mars-small128.wts"

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";


using namespace nvinfer1;

static Logger gLogger;

// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "[INFO] Loading weights map: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createMarsEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    INetworkDefinition* network = builder->createNetworkV2(0U);
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    // Create input tensor of shape { 1, 128, 64, 3} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(WEIGHT_MAP_PATH);

    // ================================== Define an empty network using TensorRT API ==================================
    // =====>>> conv1_* : conv1_1
    IConvolutionLayer* conv1_1 = network->addConvolutionNd(*data, 32, DimsHW{ 3, 3 }, weightMap["conv1_1/weights"], emptywts);
    conv1_1->setPaddingNd(DimsHW{ 1, 1 });
    assert(conv1_1);
    IScaleLayer* conv1_1_bn = addBatchNorm2d(network, weightMap, *conv1_1->getOutput(0), "conv1_1/conv1_1/bn", 1e-3); // esp = 1e-5 ? thay doi tuy theo sai so dau ra mong muon
    assert(conv1_1_bn);
    IActivationLayer* conv1_1_elu = network->addActivation(*conv1_1_bn->getOutput(0), ActivationType::kELU);
    assert(conv1_1_elu);
    // =====>>> conv1_* : conv1_2
    IConvolutionLayer* conv1_2 = network->addConvolutionNd(*conv1_1_elu->getOutput(0), 32, DimsHW{ 3, 3 }, weightMap["conv1_2/weights"], emptywts);
    conv1_2->setPaddingNd(DimsHW{ 1, 1 });
    assert(conv1_2);
    IScaleLayer* conv1_2_bn = addBatchNorm2d(network, weightMap, *conv1_2->getOutput(0), "conv1_2/conv1_2/bn", 1e-3); // esp = 1e-5 ? thay doi tuy theo sai so dau ra mong muon
    assert(conv1_2_bn);
    IActivationLayer* conv1_2_elu = network->addActivation(*conv1_2_bn->getOutput(0), ActivationType::kELU);
    assert(conv1_2_elu);
    // add MaxPool layer
    IPoolingLayer* pool1 = network->addPoolingNd(*conv1_2_elu->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setPaddingNd(DimsHW{1, 1});
    pool1->setStrideNd(DimsHW{2, 2});

    // =====>>> conv2_*
    auto conv2_1 = residualBlock(network, weightMap, *pool1->getOutput(0), 32, "conv2_1", false, true);
    auto conv2_3 = residualBlock(network, weightMap, *conv2_1->getOutput(0), 32, "conv2_3", false, false);
    // =====>>> conv3_*
    auto conv3_1 = residualBlock(network, weightMap, *conv2_3->getOutput(0), 64, "conv3_1", true, false);
    auto conv3_3 = residualBlock(network, weightMap, *conv3_1->getOutput(0), 64, "conv3_3", false, false);
    // =====>>> conv4_*
    auto conv4_1 = residualBlock(network, weightMap, *conv3_3->getOutput(0), 128, "conv4_1", true, false);
    auto conv4_3 = residualBlock(network, weightMap, *conv4_1->getOutput(0), 128, "conv4_3", false, false);
    // =====>>> fully connected layer
    IFullyConnectedLayer* fc1 = network->addFullyConnected(*conv4_3->getOutput(0), 128, weightMap["fc1/weights"], emptywts);
    assert(fc1);
    IScaleLayer* fc1_bn = addBatchNorm2d(network, weightMap, *fc1->getOutput(0), "fc1/fc1/bn", 1e-3); // esp = 1e-5 ? thay doi tuy theo sai so dau ra mong muon
    assert(fc1_bn);
    IActivationLayer* fc1_elu = network->addActivation(*fc1_bn->getOutput(0), ActivationType::kELU);
    assert(fc1_elu);
    
    auto prob = fc1_elu;

    prob->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*prob->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 20);
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createMarsEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./mars -s   // serialize model to plan file" << std::endl;
        std::cerr << "./mars -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("mars.engine", std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("mars.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        return -1;
    }


    // Subtract mean from image
    static float data[3 * INPUT_H * INPUT_W];
    for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
        data[i] = 1;

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // Run inference
    float prob[OUTPUT_SIZE];
    for (int i = 0; i < 1000; i++) {
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, 1);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    std::cout << "\nOutput:\n\n";
    for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    {
        std::cout << prob[i] << ", ";
    }
    std::cout << std::endl;

    return 0;
}