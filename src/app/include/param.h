#ifndef _PARAM_H_
#define _PARAM_H_

#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include "utility.h"


class DeepSortParam
{
    public:
        DeepSortParam() { ; }
        DeepSortParam(const std::string& filename);
        const std::string& getVideoPath() const;
        const VF_COORDINATES& getVFCoords() const;
        const std::string& getDetectionTrtEnginePath() const;
        const std::string& getDetectionModelType() const;
        const std::string& getDeepsortTrtEnginePath() const;
        const float& args_nn_budget() const;
        const float& args_max_cosine_distance() const;
        const float& dt() const;
        const float& max_iou_distance() const;
        const int& max_age() const;
        const int& n_init() const;
        const std::vector<std::string>& classes() const;
        const bool& show_detections() const;
        const void print() const;
    private:
        std::string video_path;
        std::string detection_trt_engine_path;
        std::string detection_model_type;
        std::string deepsort_trt_engine_path;
        VF_COORDINATES vf_coords;
        float args_nn_budget_value;
        float args_max_cosine_distance_value;
        float dt_value;
        float max_iou_distance_value;
        int max_age_value;
        int n_init_value;
        bool show_detection_value;
        std::vector<std::string> detection_classes;
};


#endif