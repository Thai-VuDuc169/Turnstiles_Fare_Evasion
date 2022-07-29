#include "detection.h"


Yolov5::Yolov5(const DeepSortParam &param, Logger *gLogger) 
      : Detection(const DeepSortParam &param, Logger *gLogger)
{
   assert(this->engine->getNbBindings() == 2);
   // In order to bind the buffers, we need to know the names of the input and output tensors.
   // Note that indices are guaranteed to be less than IEngine::getNbBindings()
   this->input_idx = engine->getBindingIndex(this->INPUT_BLOB_NAME);
   this->output_idx = engine->getBindingIndex(this->OUTPUT_BLOB_NAME);
   assert(this->input_idx == 0);
   assert(this->output_idx == 1);
   // Create GPU buffers on device
   CUDA_CHECK(cudaMalloc((void**)&this->buffers[this->input_idx], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
   CUDA_CHECK(cudaMalloc((void**)&this->buffers[this->output_idx], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
   // prepare input data cache in pinned memory 
   CUDA_CHECK(cudaMallocHost((void**)&this->img_host, this->MAX_IMAGE_INPUT_SIZE_THRESH * 3));
   // prepare input data cache in device memory
   CUDA_CHECK(cudaMalloc((void**)&this->img_device, this->MAX_IMAGE_INPUT_SIZE_THRESH * 3));
}

Yolov5::~Yolov5()
{
   CUDA_CHECK(cudaFree(this->buffers[this->input_idx]));
   CUDA_CHECK(cudaFree(this->buffers[this->output_idx]));
   CUDA_CHECK(cudaFree(this->img_device));
   CUDA_CHECK(cudaFreeHost(this->img_host));
}

void Yolov5::_doInference()
{
   // infer on the batch asynchronously, and DMA output back to host
   this->context.enqueue(this->BATCH_SIZE, (void **)this->buffers, this->stream, nullptr);
   CUDA_CHECK(cudaMemcpyAsync(this->prob, this->buffers[1], this->BATCH_SIZE * this->OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, this->stream));
   cudaStreamSynchronize(this->stream);
}

bool Yolov5::getFrameDetections(cv::Mat& img) override
{
   size_t size_image = img.cols * img.rows * 3;
   size_t size_image_dst = this->INPUT_H * this->INPUT_W * 3;
   //copy data to pinned memory
   memcpy(this->img_host, img.data, size_image);
   //copy data to device memory
   CUDA_CHECK(cudaMemcpyAsync(this->img_device, this->img_host, size_image, cudaMemcpyHostToDevice, this->stream));
   // dfgdf
   _preProcessing(img.cols, img.rows);
   _doInference();
   std::vector<Yolov5::HumanDetec> objs;
   _nms(objs);

   // extract rect coords from the original img
   for (size_t j = 0; j < res.size(); j++) 
   {
      cv::Rect r = get_rect(img, res[j].bbox);
      cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
      cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
   }
   cv::imwrite("results.jpg", img);
   return 1;
}

static cv::Rect get_rect(cv::Mat& img, float bbox[4]) 
{
   float l, r, t, b;
   float r_w = Yolo::INPUT_W / (img.cols * 1.0);
   float r_h = Yolo::INPUT_H / (img.rows * 1.0);
   if (r_h > r_w) {
       l = bbox[0] - bbox[2] / 2.f;
       r = bbox[0] + bbox[2] / 2.f;
       t = bbox[1] - bbox[3] / 2.f - (Yolo::INPUT_H - r_w * img.rows) / 2;
       b = bbox[1] + bbox[3] / 2.f - (Yolo::INPUT_H - r_w * img.rows) / 2;
       l = l / r_w;
       r = r / r_w;
       t = t / r_w;
       b = b / r_w;
   } else {
       l = bbox[0] - bbox[2] / 2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
       r = bbox[0] + bbox[2] / 2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
       t = bbox[1] - bbox[3] / 2.f;
       b = bbox[1] + bbox[3] / 2.f;
       l = l / r_h;
       r = r / r_h;
       t = t / r_h;
       b = b / r_h;
   }
   return cv::Rect(round(l), round(t), round(r - l), round(b - t));
}

static struct AffineMatrix{
   float value[6];
};

static bool cmp(const Yolo::HumanDetec& a, const Yolo::HumanDetec& b) {
   return a.conf > b.conf;
}

void Yolov5::_nms(std::vector<Yolov5::HumanDetec>& res)
{
   int det_size = sizeof(Yolov5::HumanDetec) / sizeof(float);
   std::map<float, std::vector<Yolov5::HumanDetec>> m;
   for (int i = 0; i < this->prob[0] && i < this->MAX_OUTPUT_BBOX_COUNT; i++) {
      if (this->prob[1 + det_size * i + 4] <= this->CONF_THRESH) continue;
      Yolov5::HumanDetec det;
      memcpy(&det, &this->prob[1 + det_size * i], det_size * sizeof(float));
      if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolov5::HumanDetec>());
      m[det.class_id].push_back(det);
   }
   for (auto it = m.begin(); it != m.end(); it++) {
      //std::cout << it->second[0].class_id << " --- " << std::endl;
      auto& dets = it->second;
      std::sort(dets.begin(), dets.end(), cmp);
      for (size_t m = 0; m < dets.size(); ++m) {
         auto& item = dets[m];
         res.push_back(item);
         for (size_t n = m + 1; n < dets.size(); ++n) {
               if (iou(item.bbox, dets[n].bbox) > this->NMS_THRESH) {
                  dets.erase(dets.begin() + n);
                  --n;
               }
         }
      }
   }
};

void Yolov5::_preProcessing(int src_width, int src_height)
{
   AffineMatrix s2d,d2s;
   float scale = std::min(this->INPUT_H / (float)src_height, this->INPUT_W / (float)src_width);

   s2d.value[0] = scale;
   s2d.value[1] = 0;
   s2d.value[2] = -scale * src_width  * 0.5  + this->INPUT_W * 0.5;
   s2d.value[3] = 0;
   s2d.value[4] = scale;
   s2d.value[5] = -scale * src_height * 0.5 + this->INPUT_H * 0.5;

   cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
   cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
   cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);

   memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

   int jobs = this->INPUT_H * this->INPUT_W;
   int threads = 256;
   int blocks = ceil(jobs / (float)threads);
   warpaffine_kernel<<<blocks, threads, 0, stream>>>(
      this->img_device, src_width*3, src_width,
      src_height, reinterpret_cast<float*>(this->buffers[this->input_idx]), this->INPUT_W,
      this->INPUT_H, 128, d2s, jobs);
}


static __global__ void warpaffine_kernel (uint8_t* src, int src_line_size, int src_width, 
                                             int src_height, float* dst, int dst_width, 
                                             int dst_height, uint8_t const_value_st,
                                             AffineMatrix d2s, int edge) 
{
   int position = blockDim.x * blockIdx.x + threadIdx.x;
   if (position >= edge) return;

   float m_x1 = d2s.value[0];
   float m_y1 = d2s.value[1];
   float m_z1 = d2s.value[2];
   float m_x2 = d2s.value[3];
   float m_y2 = d2s.value[4];
   float m_z2 = d2s.value[5];

   int dx = position % dst_width;
   int dy = position / dst_width;
   float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;
   float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;
   float c0, c1, c2;

   if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
      // out of range
      c0 = const_value_st;
      c1 = const_value_st;
      c2 = const_value_st;
   } else {
      int y_low = floorf(src_y);
      int x_low = floorf(src_x);
      int y_high = y_low + 1;
      int x_high = x_low + 1;

      uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
      float ly = src_y - y_low;
      float lx = src_x - x_low;
      float hy = 1 - ly;
      float hx = 1 - lx;
      float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
      uint8_t* v1 = const_value;
      uint8_t* v2 = const_value;
      uint8_t* v3 = const_value;
      uint8_t* v4 = const_value;

      if (y_low >= 0) {
         if (x_low >= 0)
            v1 = src + y_low * src_line_size + x_low * 3;

         if (x_high < src_width)
            v2 = src + y_low * src_line_size + x_high * 3;
      }

      if (y_high < src_height) {
         if (x_low >= 0)
            v3 = src + y_high * src_line_size + x_low * 3;

         if (x_high < src_width)
            v4 = src + y_high * src_line_size + x_high * 3;
      }

      c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
      c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
      c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
   }

   //bgr to rgb 
   float t = c2;
   c2 = c0;
   c0 = t;

   //normalization
   c0 = c0 / 255.0f;
   c1 = c1 / 255.0f;
   c2 = c2 / 255.0f;

   //rgbrgbrgb to rrrgggbbb
   int area = dst_width * dst_height;
   float* pdst_c0 = dst + dy * dst_width + dx;
   float* pdst_c1 = pdst_c0 + area;
   float* pdst_c2 = pdst_c1 + area;
   *pdst_c0 = c0;
   *pdst_c1 = c1;
   *pdst_c2 = c2;
}