#include "detection.h"

Detection::Detection(const DeepSortParam &param, Logger *gLogger)
{
   // deserialize the .engine file
   std::ifstream file(param.getDetectionTrtEnginePath(), std::ios::binary);
   if (!file.good()) {
      std::cerr << "read " << param.getDetectionTrtEnginePath() << " error!" << std::endl;
      exit(-1);
   }
   char *trtModelStream = nullptr;
   size_t size = 0;
   file.seekg(0, file.end);
   size = file.tellg();
   file.seekg(0, file.beg);
   trtModelStream = new char[size];
   assert(trtModelStream);
   file.read(trtModelStream, size);
   file.close();
   this->runtime = createInferRuntime(*gLogger);
   assert(this->runtime != nullptr);
   this->engine = runtime->deserializeCudaEngine(trtModelStream, size);
   assert(this->engine != nullptr);
   this->context = engine->createExecutionContext();
   assert(context != nullptr);
   delete[] trtModelStream;
   // create a stream to perform asyn cuda funcs
   CUDA_CHECK(cudaStreamCreate(&(this->stream)));
};

Detection::~Detection()
{
   // Release stream and buffers
   cudaStreamDestroy(this->stream);
   // Destroy the engine
   this->context->destroy();
   this->engine->destroy();
   this->runtime->destroy(); 
};

