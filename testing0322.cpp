#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

static float e = 10e-4;
static float Pi = 3.141592;
static vector<int> joints_hierarchy;

void change(vector<int>& kk) {
    kk[0] = 2;
    kk[6] = 5;
}
//  

void  static Hierarchy_set() {
    joints_hierarchy.assign(28, 0);
    joints_hierarchy[0] = -1;
    joints_hierarchy[1] = 0, joints_hierarchy[18] = 0; joints_hierarchy[22] = 0; joints_hierarchy[2] = 1;//spine
    joints_hierarchy[3] = 2; joints_hierarchy[26] = 3; joints_hierarchy[27] = 26; //neck
    joints_hierarchy[19] = 18; joints_hierarchy[20] = 19; joints_hierarchy[21] = 20;  //left leg
    joints_hierarchy[23] = 22; joints_hierarchy[24] = 23; joints_hierarchy[25] = 24;  //right leg
    joints_hierarchy[4] = 2; joints_hierarchy[5] = 4; joints_hierarchy[6] = 5; joints_hierarchy[7] = 6;  joints_hierarchy[8] = 7;// left arm
    joints_hierarchy[11] = 2; joints_hierarchy[12] = 11; joints_hierarchy[13] = 12; joints_hierarchy[14] = 13; joints_hierarchy[15] = 14; // right arm
}

int msize[2] = { 24, 3 }; //100
Mat p_set(2, msize, CV_32FC1, Scalar(0));

void make_vector(int index, Mat& vectors) {
    cout << "index : " << index << " parnet index : " << joints_hierarchy[index] << '\n';
    float x_dif = p_set.at<float>(index, 0) - p_set.at<float>(joints_hierarchy[index], 0);
    float y_dif = p_set.at<float>(index, 1) - p_set.at<float>(joints_hierarchy[index], 1);
    float z_dif = p_set.at<float>(index, 2) - p_set.at<float>(joints_hierarchy[index], 2);
    float div = sqrt(pow(x_dif, 2) + pow(y_dif, 2) + pow(z_dif, 2));
    //cout << x_dif << " " << y_dif << " " << z_dif << '\n';


    vectors.at<float>(index, 0) = x_dif / div;
    vectors.at<float>(index, 1) = y_dif / div;
    vectors.at<float>(index, 2) = z_dif / div;


}

int main2() {


    Hierarchy_set();
    Mat v_set(2, msize, CV_32FC1, Scalar(0));


    //3, 2, 1 , ���� ���
    p_set.at<float>(3, 0) = 1;
    p_set.at<float>(3, 1) = 1;
    p_set.at<float>(3, 2) = 1;

    p_set.at<float>(2, 0) = 3;
    p_set.at<float>(2, 1) = 3;
    p_set.at<float>(2, 2) = 3;

    p_set.at<float>(1, 0) = 10;
    p_set.at<float>(1, 1) = 15;
    p_set.at<float>(1, 2) = 20;

    p_set.at<float>(0, 0) = 150;
    p_set.at<float>(0, 1) = 100;
    p_set.at<float>(0, 2) = 50;

    make_vector(3, v_set);
    make_vector(2, v_set);
    cout << v_set.at<float>(3, 0) << "  " << v_set.at<float>(3, 1) << " " << v_set.at<float>(3, 2) << '\n';
    cout << v_set.at<float>(2, 0) << "  " << v_set.at<float>(2, 1) << " " << v_set.at<float>(2, 2) << '\n';
    //cout << v_set.row(3)<< '\n';
    //cout << v_set.row(2) << '\n';
    Mat c1 = v_set.row(3);
    Mat c2 = v_set.row(2);
    cout << "cross : " << c1.cross(c2) << '\n';
    float cos1 = c1.dot(c2);
    float rad1 = acos(cos1);
    cout << " cos : " << cos1 << '\n';
    cout << " rad : " << rad1 << '\n';
    cout << " theta : " << rad1 * 180 / Pi << '\n';
    //cout << c << '\n';
    // 
    // 
    ////1, 0 , -1 ���� ���
    cout << '\n';
    v_set.at<float>(0, 0) = 0;
    v_set.at<float>(0, 1) = 1;
    v_set.at<float>(0, 2) = 0;
    make_vector(1, v_set);

    cout << v_set.at<float>(0, 0) << "  " << v_set.at<float>(0, 1) << " " << v_set.at<float>(0, 2) << '\n';
    cout << v_set.at<float>(1, 0) << "  " << v_set.at<float>(1, 1) << " " << v_set.at<float>(1, 2) << '\n';


    Mat c3 = v_set.row(1);
    Mat c4 = v_set.row(0);

    cout << "cross : " << c4.cross(c3) << '\n';
    float cos2 = c3.dot(c4);
    float rad2 = acos(cos2);
    cout << " cos : " << cos2 << '\n';
    cout << " rad : " << rad2 << '\n';
    cout << " theta : " << rad2 * 180 / Pi << '\n';

    Mat test;
    Rodrigues(c4.cross(c3) * rad2, test);
    Mat test2;
    Rodrigues(test, test2);
    cout << test << '\n';
    cout << test2 << '\n';

    cout << "inv: " << test.inv() << '\n';;
    return 0;

}