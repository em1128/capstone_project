#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

int main2() {
   
    int out[3];
    out[0] = 1;
    out[1] = 2;
    out[2] = 3;

    cv::Mat M(3, out, CV_32FC1, cv::Scalar(0));


    float ain[9] = { 0,1,2,3,4,5,6,7,8 };
    //M.ptr<float>(0)=new cv::Mat(2,3, CV_32FC1,ain);
    float M_res = 0;
    
    // atÀ» »ç¿ë 
    /*
    for (int loopz = 0; loopz < out[2]; loopz++) {
        std::cout << std::endl << std::endl << "Dimension " << loopz << std::endl;
        for (int loopy = 0; loopy < out[0]; loopy++) {
            std::cout << std::endl;
            for (int loopx = 0; loopx < out[1]; loopx++) {
                M_res = M.at<float>(loopy, loopx, loopz);
                std::cout << M_res << " ";
            }
        }
    }
    */
    
    std::cout << "Total size " << M.size << " and, for example, rows " << M.size[0] << std::endl;

    std::cout << "IS the matrix really continuous? " << M.isContinuous() << " Yes" << std::endl;
    int counter = 0;
    for (int i = 0; i < M.size[0]; i++) {
        std::cout << i << "inum" << std::endl;
        for (int j = 0; j < M.size[1]; j++) {
            float* p = M.ptr<float>(i,j);
            for (int k = 0; k < M.size[2]; k++) {
                std::cout << *(p + k) << " ";
                counter += 1;
            }

            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << "Number of elements seen " << counter << " matrix number of elements " << 3*6*9 << std::endl;


	//::Rodrigues(rvec, rmax);
    
    
	return 0;
}