# Turnstiles_Fare_Evasion
This project was created to deal with the problem of detecting fare evasion through turnstiles in public areas such as subway stations, airports, etc. The punishment for “fare evasion” is on par with that of property theft, driving under the influence, or assault. Therefore, I want to do this project to prevent this problem in places where there is no human's control, increase more revenue in the e-ticketing industry.

## Requirements

* OpenCV (>= 3.4)
* Eigen3
* Boost
* CUDA (>= 10.0)
* Tensorflow ( = 1.14)

## Tips 
- Initialization can take a lot of time because TensorRT tries to find out the best and faster way to perform your network on your platform. To do it only once and then use the already created engine you can serialize your engine. Serialized engines are not portable across different GPU models, platforms, or TensorRT versions. Engines are specific to the exact hardware and software they were built on. More info can be found here:
- "dpkg -l | grep nvinfer" : This command to display version of TensorRT in the current hardware