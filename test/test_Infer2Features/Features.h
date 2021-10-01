#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


typedef unsigned char uint8;

class FeatureTensor
{
public:
	// Constructors
	FeatureTensor();
   // Destructors
	~FeatureTensor();
   
	FeatureTensor& operator = (const FeatureTensor&);
	bool getRectsFeature(const cv::Mat& img, DETECTIONS& d);

private:
	bool init();
	void tobuffer(const std::vector<cv::Mat> &imgs, uint8 *buf);
	int feature_dim;
	std::shared_ptr<tensorflow::Session> session;
	std::vector<tensorflow::Tensor> output_tensors;
	std::vector<tensorflow::string> outnames;
	tensorflow::string input_layer;
	std::string tf_model_meta;
	std::string tf_model_data;
};