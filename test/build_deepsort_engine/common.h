#ifndef _MARS_COMMON_H_
#define _MARS_COMMON_H_

// #include <fstream>
#include <map>
#include <cmath>
// #include <sstream>
// #include <vector>
// #include <opencv2/opencv.hpp>
#include <cassert>
#include "NvInfer.h"

using namespace nvinfer1;

IScaleLayer* addBatchNorm2d(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps)
{
    // float* gamma = (float*)weightMap[lname + "gamma"].values; // scale
    float gamma = 1.0; // const scale
    float* beta = (float*)weightMap[lname + "/beta"].values;   // offset
    float* mean = (float*)weightMap[lname + "/moving_mean"].values;
    float* var = (float*)weightMap[lname + "/moving_variance"].values;
    int len = weightMap[lname + "/moving_variance"].count;

    float* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (auto i = 0; i < len; i++)
    {
        scval[i] = gamma / sqrt(var[i] + eps);
    }
    Weights scale{ DataType::kFLOAT, scval, len };

    float* shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (auto i = 0; i < len; i++)
    {
        shval[i] = beta[i] - mean[i] * gamma / sqrt(var[i] + eps);
    }
    Weights shift{ DataType::kFLOAT, shval, len };

    float* pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (auto i = 0; i < len; i++)
    {
        pval[i] = 1.0;
    }
    Weights power{ DataType::kFLOAT, pval, len };

    weightMap[lname + "/scale"] = scale;
    weightMap[lname + "/shift"] = shift;
    weightMap[lname + "/power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer* innerBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, std::string lname, bool increase_dim= true)
{
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    int out_features = outch;
    int stride = 1;
    if (increase_dim)
        stride = 2;
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, out_features, DimsHW{3, 3}, weightMap[lname + "/1/weights"], emptywts);
    conv1->setPaddingNd(DimsHW{ 1, 1 });
    conv1->setStrideNd(DimsHW{stride, stride});
    assert(conv1);
    IScaleLayer* conv1_bn = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "/1/" + lname + "/1/bn" , 1e-3); // esp = 1e-5 ? thay doi tuy theo dau ra mong muon
    assert(conv1_bn);
    IActivationLayer* conv1_elu = network->addActivation(*conv1_bn->getOutput(0), ActivationType::kELU);
    assert(conv1_elu);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*conv1_elu->getOutput(0), out_features, DimsHW{3, 3}, weightMap[lname + "/2/weights"], weightMap[lname + "/2/biases"]);
    conv2->setPaddingNd(DimsHW{ 1, 1 });
    assert(conv2);
    return conv2;
}

ILayer* residualBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, std::string lname, bool increase_dim = false, bool is_first = false) 
{
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    ITensor *pre_block_network = &input;
    if (!is_first)
    //     pre_block_network = input;
    // else
    {
        IScaleLayer* conv1_pre_bn = addBatchNorm2d(network, weightMap, input, lname + "/bn" , 1e-3); // esp = 1e-5 ? thay doi tuy theo dau ra mong muon
        assert(conv1_pre_bn);
        IActivationLayer* conv1_pre_elu = network->addActivation(*conv1_pre_bn->getOutput(0), ActivationType::kELU);
        assert(conv1_pre_elu);
        pre_block_network = conv1_pre_elu->getOutput(0);
    }
    
    auto innner_block = innerBlock(network, weightMap, *pre_block_network, outch, lname, increase_dim);
    IElementWiseLayer* element_wise;
    if (increase_dim)
    {
        auto projection_block = network->addConvolutionNd(input, outch, DimsHW{1, 1}, weightMap[lname + "/projection/weights"], emptywts);
        projection_block->setStrideNd(DimsHW{2, 2});
        assert(projection_block);
        element_wise = network->addElementWise(*innner_block->getOutput(0), *projection_block->getOutput(0), ElementWiseOperation::kSUM);
        assert(element_wise);
    }
    else
    {
        element_wise = network->addElementWise(*innner_block->getOutput(0), input, ElementWiseOperation::kSUM);
        assert(element_wise);
    }
    return element_wise;
}

#endif