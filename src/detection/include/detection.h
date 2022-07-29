#ifndef _DETECTION_H_
#define _DETECTION_H_

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <map>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cstdint>
#include "param.h"
#include "logging.h"

using namespace nvinfer1;

class Detection
{
   protected:
      cudaStream_t stream;
      IRuntime* runtime;
      ICudaEngine* engine;
      IExecutionContext* context;
      
   public:
      Detection(const DeepSortParam &param, Logger *gLogger);
      virtual ~Detection(); // If you're using polymorphism and your derived instance is pointed to by a base class pointer, then the derived class destructor is only called if the base destructor is virtual.
      virtual bool getFrameDetections(cv::Mat& frame) = 0;
};

class Yolov5 : public Detection
{
   private:
      struct alignas(float) HumanDetec 
      {
         //center_x center_y w h
         float bbox[4];
         float conf;  // bbox_conf * cls_conf
         float class_id;
      };

   private:
      const float NMS_THRESH = 0.4;
      const float CONF_THRESH = 0.5;
      const uint MAX_IMAGE_INPUT_SIZE_THRESH = 3000 * 3000;
      const int INPUT_H = 640;
      const int INPUT_W = 640;
      const int BATCH_SIZE = 1;
      const int MAX_OUTPUT_BBOX_COUNT = 1000;
      const int OUTPUT_SIZE = 6001; // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
      const char* INPUT_BLOB_NAME = "data";
      const char* OUTPUT_BLOB_NAME = "prob";
      float prob[6001];
      int input_idx;
      int output_idx;
      float* buffers[2];
      uint8_t* img_host;
      uint8_t* img_device;

   public:
      Yolov5(const DeepSortParam &param, Logger *gLogger);
      virtual ~Yolov5();
      bool getFrameDetections(cv::Mat& frame) override;
   private:
      void _doInference();
      void _preProcessing(int src_width, int src_height);
      void _nms(std::vector<Yolov5::Detection>& res);
};

#endif