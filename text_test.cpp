#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


using namespace cv;
using namespace std;

int main()
{
    float data[6][3] = {
        {255, 255, 255},{1,1,1},
        {58, 106, 178},{168, 25, 132},
        {168, 25, 132}, {58, 106, 178},
    };
    // size(width, height)
    Mat img(Size(2, 3), CV_32FC3, data);

    cout << img << endl;

    img.convertTo(img, CV_8UC3);
    
    namedWindow("img", WINDOW_FREERATIO);
    
    cv::imshow("img", img);
    cout << "what is problem?" << '\n';
    waitKey(0);

}
