#include "param.h"

DeepSortParam::DeepSortParam(const std::string& filename)
{
  std::ifstream file;
  
  try
  {
    file.open(filename);
  }
  catch(...)
  {
    std::cerr << "Cannot open " << filename << std::endl;
    file.close();
    exit(-1);
  }
  
  if(!file.is_open())
  {
    std::cerr << "Error: file " << filename << " not found!" << std::endl;
    exit(-1);
  }
  
  std::string line;
  while(std::getline(file, line))
  {
      std::remove_if(line.begin(), line.end(), isspace);
      if(line.empty())
      {
	    continue;
      }
      else if(line.find("[VIDEO_PATH]") != std::string::npos)
      {
        std::getline(file, line);
        try
        {
            this->video_path = line;
        }
        catch(...)
        {
            std::cerr << "Error in converting the VIDEO_PATH: " << line << std::endl;
            exit(-1);
        }
      }
      else if(line.find("[VIRTUAL_FENCE_COORDINATES]") != std::string::npos)
      {
        std::getline(file, line);
        try
        {
            int ax, ay, bx, by;
            std::istringstream iss(line);
            iss >> ax >> ay >> bx >> by;
            this->vf_coords = VF_COORDINATES(ax, ay, bx, by);
        }
        catch(...)
        {
            std::cerr << "Error in converting the VIRTUAL_FENCE_COORDINATES: " << line << std::endl;
            exit(-1);
        }
      }
      else if(line.find("[DETECTION_TRT_ENGINE_PATH]") != std::string::npos)
      {
        std::getline(file, line);
        try
        {
            this->detection_trt_engine_path = line;
        }
        catch(...)
        {
            std::cerr << "Error in converting the DETECTION_TRT_ENGINE_PATH: " << line << std::endl;
            exit(-1);
        }
      }
      else if(line.find("[DETECTION_MODEL_TYPE]") != std::string::npos)
      {
        std::getline(file, line);
        try
        {
            this->detection_model_type = line;
        }
        catch(...)
        {
            std::cerr << "Error in converting the DETECTION_MODEL_TYPE: " << line << std::endl;
            exit(-1);
        }
      }
      else if(line.find("[DEEPSORT_TRT_ENGINE_PATH]") != std::string::npos)
      {
        std::getline(file, line);
        try
        {
            this->deepsort_trt_engine_path = line;
        }
        catch(...)
        {
            std::cerr << "Error in converting the DEEPSORT_TRT_ENGINE_PATH: " << line << std::endl;
            exit(-1);
        }
      }
      else if(line.find("[ARGS_NN_BUDGET]") != std::string::npos)
      {
        std::getline(file, line);
        try
        {
            args_nn_budget_value = atof(line.c_str());
        }
        catch(...)
        {
            std::cerr << "Error in converting the ARGS_NN_BUDGET: " << line << std::endl;
            exit(-1);
        }
      }
      else if(line.find("[ARGS_MAX_COSINE_DISTANCE]") != std::string::npos)
      {
        std::getline(file, line);
        try
        {
            args_max_cosine_distance_value = atof(line.c_str());
        }
        catch(...)
        {
            std::cerr << "Error in converting the ARGS_MAX_COSINE_DISTANCE: " << line << std::endl;
            exit(-1);
        }
      }
      else if(line.find("[DT]") != std::string::npos)
      {
        std::getline(file, line);
        try
        {
            dt_value = atof(line.c_str());
        }
        catch(...)
        {
            std::cerr << "Error in converting the DT: " << line << std::endl;
            exit(-1);
        }
      }
      else if(line.find("[MAX_IOU_DISTANCE]") != std::string::npos)
      {
        std::getline(file, line);
        try
        {
            max_iou_distance_value = atof(line.c_str());
        }
        catch(...)
        {
            std::cerr << "Error in converting the MAX_IOU_DISTANCE: " << line << std::endl;
            exit(-1);
        }
      }
      else if(line.find("[MAX_AGE]") != std::string::npos)
      {
        std::getline(file, line);
        try
        {
            max_age_value = atoi(line.c_str());
        }
        catch(...)
        {
            std::cerr << "Error in converting the MAX_AGE: " << line << std::endl;
            exit(-1);
        }
      }
      else if(line.find("[N_INIT]") != std::string::npos)
      {
        std::getline(file, line);
        try
        {
            n_init_value = atoi(line.c_str());
        }
        catch(...)
        {
            std::cerr << "Error in converting the N_INIT: " << line << std::endl;
            exit(-1);
        }
      }
      else if(line.find("[SHOW_DETECTIONS]") != std::string::npos)
      {
        std::getline(file, line);
        try
        {
            show_detection_value = atoi(line.c_str());
        }
        catch(...)
        {
            std::cerr << "Error in converting the SHOW_DETECTIONS: " << line << std::endl;
            exit(-1);
        }
      }
      else if(line.find("[CLASSES]") != std::string::npos)
      {
        std::getline(file, line);
        try
        {
            std::ifstream class_file;
            try
            {
                class_file.open(line);
            }
            catch(...)
            {
                std::cerr << "Cannot open " << line << std::endl;
                file.close();
                exit(-1);
            }
            
            if(!class_file.is_open())
            {
                std::cerr << "Error: file " << line << " not found!" << std::endl;
                exit(-1);
            }
            
            
            std::string class_line;
            while(std::getline(class_file, class_line))
            {
                detection_classes.push_back(class_line);
            }

            class_file.close();
        }
        catch(...)
        {
            std::cerr << "Error in converting the N_INIT: " << line << std::endl;
            exit(-1);
        }
      }
  }
 
  file.close();
  
  print();
}

const std::string& DeepSortParam::getVideoPath() const
{
    return this->video_path;
}
const VF_COORDINATES& DeepSortParam::getVFCoords() const
{
    return this->vf_coords;
}
const std::string& DeepSortParam::getDetectionTrtEnginePath() const
{
    return this->detection_trt_engine_path;
}
const std::string& DeepSortParam::getDetectionModelType() const
{
    return this->detection_model_type;
}
const std::string& DeepSortParam::getDeepsortTrtEnginePath() const
{
    return this->deepsort_trt_engine_path;
}
const float& DeepSortParam::args_nn_budget() const
{
    return args_nn_budget_value;
}
const float& DeepSortParam::args_max_cosine_distance() const
{
    return args_max_cosine_distance_value;
}
const float& DeepSortParam::dt() const
{
    return dt_value;
}
const float& DeepSortParam::max_iou_distance() const
{
    return max_iou_distance_value;
}
const int& DeepSortParam::max_age() const
{
    return max_age_value;
}
const int& DeepSortParam::n_init() const
{
    return n_init_value;
}
const std::vector<std::string>& DeepSortParam::classes() const
{
    return detection_classes;
}
const bool& DeepSortParam::show_detections() const
{
    return show_detection_value;
}

const void DeepSortParam::print() const
{
    std::cout << "[VIDEO_PATH]: " << this->video_path << std::endl;
    std::cout << "[VIRTUAL_FENCE_COORDINATES]: " << this->vf_coords  << std::endl;
    std::cout << "[DETECTION_TRT_ENGINE_PATH]: " << this->detection_trt_engine_path << std::endl;
    std::cout << "[DETECTION_MODEL_TYPE]: " << this->detection_model_type << std::endl;
    std::cout << "[DEEPSORT_TRT_ENGINE_PATH]: " << this->deepsort_trt_engine_path << std::endl;
    std::cout << "[ARGS_NN_BUDGET]: " << args_nn_budget_value << std::endl;
    std::cout << "[ARGS_MAX_COSINE_DISTANCE]: " << args_max_cosine_distance_value << std::endl;
    std::cout << "[DT]: " << dt_value << std::endl;
    std::cout << "[MAX_IOU_DISTANCE]: " << max_iou_distance_value << std::endl;
    std::cout << "[MAX_AGE]: " << max_age_value << std::endl;
    std::cout << "[N_INIT]: "  << n_init_value << std::endl;
    if(detection_classes.size() != 0)
    {
        std::cout << "[CLASSES]: ";
        for(const auto& c : detection_classes)
        {
            std::cout << c << ", ";
        }
        std::cout << "total=" << detection_classes.size() << " classes" << std::endl;
    }
    std::cout << "[SHOW_DETECTIONS]: "  << show_detection_value << std::endl;
}
