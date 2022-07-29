#include <iostream>
// #include "VideoTracker.h"
#include "param.h"
#include "detection.h"

static Logger *gLogger = new Logger();

int main(int argc, char **argv)
{
	if(argc != 2)
	{
		std::cerr << "Usage: " << argv[0] << " /path/to/the/config/file" << std::endl;
		exit(-1);
	}

	const DeepSortParam params(argv[1]);
	Yolov5 detector (params, gLogger);

	

	// params.read(argv[1]);
	
	// params.print();	
	// VideoTracker t(params);
	// if(t.run() == false) 
	// {
	// 	std::cout << t.showErrMsg() << std::endl;
	// }
	return 0;
}
