#ifndef _UTILS_
#define _UTILS_

#include <opencv2/opencv.hpp>
#include <ostream>
#include <cuda_runtime_api.h>
// #include "dataType.h"

// class DETECTION_ROW 
// {
//     public:
//         DETECTBOX tlwh; //np.float
//         float confidence; //float
//         int class_num;
//         cv::Scalar color;
//         FEATURE feature; //np.float32
//         DETECTBOX to_xyah() const;
//         DETECTBOX to_tlbr() const;
// };
// typedef std::vector<DETECTION_ROW> DETECTIONS;

struct VF_COORDINATES
{
    struct Point {int x,y;};
    Point A;
    Point B;
    VF_COORDINATES() {;}
    VF_COORDINATES(int A_x, int A_y, int B_x, int B_y)
    {
        this->A.x = A_x;
        this->A.y = A_y;
        this->B.x = B_x;
        this->B.y = B_y;
    }
    friend std::ostream& operator<< (std::ostream &out, VF_COORDINATES const &vf_coords_temp)
    {
        out << "A= (" << vf_coords_temp.A.x << ", " << vf_coords_temp.A.y << ")   " 
            << "B= (" << vf_coords_temp.B.x << ", " << vf_coords_temp.B.y << ")";
        return out;
    }
};


#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
#endif  // CUDA_CHECK



#endif