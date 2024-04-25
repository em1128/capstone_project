#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;

static vector<int> joints_hierarchy;
void  static Hierarchy_set();

void Make_vector(int index, cv::Mat& vectors, cv::Mat& p_set);

int main() {
	Hierarchy_set();

	int vsize[2] = { 28,3 }; // vector size
	int vsize2D[2] = { 28,2 }; // 2D vector size 

	cv::Mat p_set(2, vsize, CV_32FC1, cv::Scalar(0)); //3d points set 28*3
	cv::Mat p_set_2D(2, vsize2D, CV_32FC1, cv::Scalar(0)); // vectors set 28*2
	cv::Mat grayMat(720, 1280, CV_8UC1); // gray image
	
	// load data
	cv::FileStorage fsRead("saveMat.xml", cv::FileStorage::READ);

	fsRead["grayImage"] >> grayMat;
	fsRead["joint3D"] >> p_set;
	fsRead["joint2D"] >> p_set_2D;
	fsRead.release();
	cv::Mat v_set(2, vsize, CV_32FC1, cv::Scalar(0)); // vectors set 28*3
	v_set.at<float>(0, 0) = 0; //false pelvis vector set
	v_set.at<float>(0, 1) = -1;
	v_set.at<float>(0, 2) = 0;
	
	// validate data
	for (int i = 0; i <= 27; i++) {
		if (i == 9 || i == 16 || i == 17 || i == 10 || i == 0) { continue; }
		
		Make_vector(i, v_set, p_set); //인덱스의 부모 좌표에서 인덱스 좌표로 가는 3차원 벡터
		cout << "index " << i << v_set.at<float>(i, 0) << "  " << v_set.at<float>(i, 1) << " " << v_set.at<float>(i, 2) << '\n';
	}
	
	/*
	cv::namedWindow("image"); // 이미지를 보여주기 위한 빈 창
	cv::imshow("image", grayMat); // "image"라는 이름의 창에 이미지를 넣어 보여줌
	cv::waitKey(); // 종료 키 대기
	*/
	
	return 0;
}

void  static Hierarchy_set()
{
	joints_hierarchy.assign(28, 0);
	joints_hierarchy[0] = -1;
	joints_hierarchy[1] = 0, joints_hierarchy[18] = 0; joints_hierarchy[22] = 0; joints_hierarchy[2] = 1;//spine
	joints_hierarchy[3] = 2; joints_hierarchy[26] = 3; joints_hierarchy[27] = 26; //neck
	joints_hierarchy[19] = 18; joints_hierarchy[20] = 19; joints_hierarchy[21] = 20;  //left leg
	joints_hierarchy[23] = 22; joints_hierarchy[24] = 23; joints_hierarchy[25] = 24;  //right leg
	joints_hierarchy[4] = 2; joints_hierarchy[5] = 4; joints_hierarchy[6] = 5; joints_hierarchy[7] = 6;  joints_hierarchy[8] = 7;// left arm
	joints_hierarchy[11] = 2; joints_hierarchy[12] = 11; joints_hierarchy[13] = 12; joints_hierarchy[14] = 13; joints_hierarchy[15] = 14; // right arm
}

void Make_vector(int index, cv::Mat& vectors, cv::Mat& p_set)
{
	cout << "index : " << index << " parnet index : " << joints_hierarchy[index] << '\n';
	float x_dif = p_set.at<float>(index, 0) - p_set.at<float>(joints_hierarchy[index], 0);
	float y_dif = p_set.at<float>(index, 1) - p_set.at<float>(joints_hierarchy[index], 1);
	float z_dif = p_set.at<float>(index, 2) - p_set.at<float>(joints_hierarchy[index], 2);
	float div = sqrt(pow(x_dif, 2) + pow(y_dif, 2) + pow(z_dif, 2));
	cout << x_dif << " " << y_dif << " " << z_dif << '\n';
	vectors.at<float>(index, 0) = x_dif / div;
	vectors.at<float>(index, 1) = y_dif / div;
	vectors.at<float>(index, 2) = z_dif / div;
}