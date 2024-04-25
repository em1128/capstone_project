#include <stdio.h>
#include <stdlib.h>

#include <k4abttypes.h>
#include <k4a/k4a.h>
#include <k4abt.h>
#include <iostream>
#include<vector>
#include <sstream>
#include <fstream>
#include<iostream>
#include <memory>
#include <thread>
#include "DTW.h"
#include <time.h>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"



using namespace std;
using namespace cv;

static float e = 10e-4;
static float Pi = 3.141592;

#define VERIFY(result, error)                                                                            \
    if(result != K4A_RESULT_SUCCEEDED)                                                                   \
    {                                                                                                    \
        printf("%s \n - (File: %s, Function: %s, Line: %d)\n", error, __FILE__, __FUNCTION__, __LINE__); \
        exit(1);                                                                                         \
    }                                                                                                    \

// 
//             -  18(HIP_LEFT)   -19(KNEE_RIGHT)   - 20(ANKLE_LEFT) - 21(FOOT_LEFT)
// 
//   -1(False_PELVIS) - 0(PELVIS) -  1(SPINE_NAVAL)  - 2 (SPINE_CHEST) 
//             
//              -  22(HIP_RIGHT) - 23(KNEE_RIGHT) -  24(ANKLE_RIGHT) - 25(FOOT_RIGHT)
// 
/////////////////////////////////////////////////////////////////////////////////////
// 
//           -4(CLAVICLE_LEFT) - 5(SHOULDER_LEFT) - 6(ELBOW_LEFT) - 7(WRIST_LEFT) - 8(HAND_LEFT)
// 
//  2(SPINE_CHEST)  -   3(NECK)  - 26(HEAD) - 27(NOSE)
//
//       -  11(CLAVICLE_RIGHT)   - 12(SHOULDER_RIGHT) - 13(ELBOW_RIGHT) - 14(WRIST_RIGHT) - 15(HAND_RIGHT)


//                                  27
//                                  I
//                                  26
//                                  I
//                                  3
//                                  I
//  8 - - 7 - - 6 - - 5 - - 4 - -   2  - - 11 - - 12 - - 13 - - 14 - - 15
//                                 II   
//                
//                                  1
// 
//                                  II
//                            22- - 0 - - 18
//                           //            ＼＼
//                          23               19
//                          II               II
//                          24               20
//                          II               II
//                          25               21

static vector<int> joints_hierarchy;
const char* intrinsicArr[15] = { "cx", "cy", "fx","fy","k1","k2","k3","k4","k5","k6","codx","cody","p2","p1","metric radius" };
const int vsize[2] = { 56, 3 };// vector size
const int rsize[3] = { 56, 3 ,3 }; //matrix size
// 추가 0407
const int vsize_2D[2] = { 56,2 }; // vector 2D size
Scalar colors[28];
Scalar colors0[28];


//check 2points
void Print2p(float x, float y);
//check 3points
void Print3p(float x, float y, float z);
//check vector
void vector_print(vector<vector<vector<float>>>& v);
void Print_joint_error(Mat pset1, Mat pset2, int i_body, string c);
void CheckIntrinsicParam(k4a_calibration_t sensor_calibration);

void Make_vector(int index, Mat& vectors, Mat& p_set, int i_body);
void Make_vector2D(int index, Mat& p_set, int i_body);

void Get_theta(int index, Mat& vectors, float theta[], int i_body);
void Get_n_vector(int index, Mat& vectors, Mat& nv_set, int i_body);
void Get_rotation_matrix(int index, Mat& nv_set, float th_set[], Mat& rot_set, Mat& inv_rot_set, int i_body);

static void create_xy_table(const k4a_calibration_t* calibration, k4a_image_t xy_table);


static void generate_point_cloud(const k4a_image_t depth_image,
    const k4a_image_t xy_table,
    k4a_image_t point_cloud,
    int* point_count);
static void write_point_cloud(const char* file_name, const k4a_image_t point_cloud, int point_count);
//void MakeExtrinsicMatrix(string filename1, string filename2, Mat& transform);

void static color();
void static color0();
void Multiply_extrinsic_matrix(Mat& pset1, Mat& pset2, Mat transform, int i_body);

static void Get_3dPoints(vector<k4abt_skeleton_t> skeleton, int i_body, Mat& p_set);
static void  Drawing_circle(vector<k4abt_skeleton_t> skeleton, int i_body, Mat& p_set, k4a_calibration_t sensor_calibration, k4a_float2_t xy_color[], Mat& p_set_2D, k4abt_frame_t body_frame, Mat& colorMat,
    k4a_image_t color_image, uint8_t* color_buffer, Scalar colors[]);
static void Drawing_marker(vector<k4abt_skeleton_t> skeleton, int i_body, Mat& p_set, k4a_calibration_t sensor_calibration, k4a_float2_t xy_color[], Mat& p_set_2D, k4abt_frame_t body_frame, Mat& colorMat,
    k4a_image_t color_image, uint8_t* color_buffer, Scalar colors[]);



int main()
{
    double p = 2;  // the p-norm to use; 2.0 = euclidean, 1.0 = manhattan
    std::vector<std::vector<double> > a = { {0, 2}, {1, 2}, {2, 2}, {3, 2}, {4, 2} };
    std::vector<std::vector<double> > b = { {0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0} };
    // initialize the DTW object
    DTW::DTW MyDtw(a, b, p);
    // The distance is calculated and stored in MyDtw.distance
    std::cout << "DTW distance: " << MyDtw.distance << std::endl;
    //Hierarchy_initalize
    color();
    color0();

    //save video 727
    int fourcc = VideoWriter::fourcc('M', 'J', 'P', 'G');
    VideoWriter writer;
    writer.open("video_file.avi", fourcc, 15, Size(720, 1280), 0);

    k4a_device_t device = NULL;
    /* k4a_device_t device2 = NULL;*/
    VERIFY(k4a_device_open(0, &device), "Open K4A Device 1 failed!");
    /*VERIFY(k4a_device_open(1, &device2), "Open K4A Device 2 failed!");*/

    const int32_t TIMEOUT_IN_MS = 1000000;
    // Start camera. Make sure depth camera is enabled.
    k4a_device_configuration_t deviceConfig = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    deviceConfig.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    deviceConfig.camera_fps = K4A_FRAMES_PER_SECOND_15;
    deviceConfig.depth_mode = K4A_DEPTH_MODE_NFOV_2X2BINNED;
    deviceConfig.color_resolution = K4A_COLOR_RESOLUTION_720P;

    /*k4a_device_configuration_t deviceConfig2 = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    deviceConfig2.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    deviceConfig2.camera_fps = K4A_FRAMES_PER_SECOND_15;
    deviceConfig2.depth_mode = K4A_DEPTH_MODE_NFOV_2X2BINNED;
    deviceConfig2.color_resolution = K4A_COLOR_RESOLUTION_720P;*/
    VERIFY(k4a_device_start_cameras(device, &deviceConfig), "Start K4A cameras 1 failed!");
    /* VERIFY(k4a_device_start_cameras(device2, &deviceConfig2), "Start K4A cameras 2 failed!");*/

     // sensor calibration
    k4a_calibration_t sensor_calibration;
    /* k4a_calibration_t sensor_calibration2;*/
    VERIFY(k4a_device_get_calibration(device, deviceConfig.depth_mode, deviceConfig.color_resolution, &sensor_calibration),
        "Get depth camera 1 calibration failed!");
    /* VERIFY(k4a_device_get_calibration(device2, deviceConfig2.depth_mode, deviceConfig2.color_resolution, &sensor_calibration2),
         "Get depth camera 2 calibration failed!");*/

         //Check intrinsics info
    cout << "Kinect 1 intrinsic: " << '\n';
    CheckIntrinsicParam(sensor_calibration);
    /*cout << "Kinect 2 intrinsic: " << '\n';
    CheckIntrinsicParam(sensor_calibration2);*/

    //Body traker config
    k4abt_tracker_t tracker = NULL;
    /*k4abt_tracker_t tracker2 = NULL;*/
    k4abt_tracker_configuration_t tracker_config = K4ABT_TRACKER_CONFIG_DEFAULT;
    VERIFY(k4abt_tracker_create(&sensor_calibration, tracker_config, &tracker), "Body tracker 1 initialization failed!");
    /* VERIFY(k4abt_tracker_create(&sensor_calibration2, tracker_config, &tracker2), "Body tracker2 initialization failed!");*/


     //frame start
    int frame_count = 0;
    do
    {
        double loops = clock();
        k4a_capture_t sensor_capture;
        // k4a_capture_t sensor_capture2;

        k4a_wait_result_t get_capture_result = k4a_device_get_capture(device, &sensor_capture, K4A_WAIT_INFINITE);
        //k4a_wait_result_t get_capture_result2 = k4a_device_get_capture(device2, &sensor_capture2, K4A_WAIT_INFINITE);
        double loops2 = clock();
        cout << "capture " << loops2 - loops << '\n';

        if (get_capture_result == K4A_WAIT_RESULT_SUCCEEDED/* && get_capture_result2 == K4A_WAIT_RESULT_SUCCEEDED*/)
        {
            frame_count++;

            k4a_wait_result_t queue_capture_result = k4abt_tracker_enqueue_capture(tracker, sensor_capture, K4A_WAIT_INFINITE);
            double loops3 = clock();
            cout << "body capture1 " << loops3 - loops2 << '\n';
            //k4a_wait_result_t queue_capture_result2 = k4abt_tracker_enqueue_capture(tracker2, sensor_capture2, K4A_WAIT_INFINITE);
            double loops4 = clock();
            cout << "body capture2 " << loops4 - loops3 << '\n';

            k4a_capture_release(sensor_capture);
            // k4a_capture_release(sensor_capture2);// Remember to release the sensor capture once you finish using it

            if (queue_capture_result == K4A_WAIT_RESULT_TIMEOUT /* ||queue_capture_result2 == K4A_WAIT_RESULT_TIMEOUT*/) {
                // It should never hit timeout when K4A_WAIT_INFINITE is set.
                printf("Error! Add capture to tracker process queue timeout!\n");
                continue;
            }
            else if (queue_capture_result == K4A_WAIT_RESULT_FAILED /*|| queue_capture_result2 == K4A_WAIT_RESULT_FAILED*/)
            {
                printf("Error! Add capture to tracker process queue failed!\n");
                continue;
            }


            k4abt_frame_t body_frame = NULL;
            //k4abt_frame_t body_frame2 = NULL;
            double camera1time = clock();
            k4a_wait_result_t pop_frame_result = k4abt_tracker_pop_result(tracker, &body_frame, 0);
            double camera2time = clock();
            // k4a_wait_result_t pop_frame_result2 = k4abt_tracker_pop_result(tracker2, &body_frame2, 0);
            double time2 = clock();
            cout << "ㄱㄷ1 time" << camera2time - camera1time << '\n';
            //cout << "ㄱㄷ2 time" << time2 - camera2time << '\n';

            if (pop_frame_result == K4A_WAIT_RESULT_SUCCEEDED /* && pop_frame_result2 == K4A_WAIT_RESULT_SUCCEEDED*/)
            {
                // Successfully popped the body tracking result. Start your processing
                k4a_image_t depth = k4a_capture_get_depth_image(sensor_capture);
                k4a_image_t color_image = k4a_capture_get_color_image(sensor_capture);
                size_t num_bodies = k4abt_frame_get_num_bodies(body_frame);
                /*
                k4a_image_t depth2 = k4a_capture_get_depth_image(sensor_capture2);
                k4a_image_t color_image2 = k4a_capture_get_color_image(sensor_capture2);
                size_t num_bodies2 = k4abt_frame_get_num_bodies(body_frame2);
                */
                /* printf("%zu bodies are detected!\n", num_bodies);*/
                if (1/*num_bodies == num_bodies2 *//*&& num_bodies != 0*/) {
                    uint8_t* color_buffer = k4a_image_get_buffer(color_image);
                    //uint8_t* color_buffer2 = k4a_image_get_buffer(color_image2);

                    //cv::namedWindow("image"); // 이미지를 보여주기 위한 빈 창
                    cv::Mat colorMat(720, 1280, CV_8UC4, (void*)color_buffer, cv::Mat::AUTO_STEP);
                    //cv::Mat colorMat2(720, 1280, CV_8UC4, (void*)color_buffer2, cv::Mat::AUTO_STEP);// color camera 이미지

                    //cv::Mat videoframe;
                    //VideoCapture cap(0);
                    //cap >> videoframe;
                    //hconcat(colorMat, colorMat2, colorMat);
                    //resize(colorMat, colorMat, Size(1280, 360));
                    resize(colorMat, colorMat, Size(720, 1280), 0, 0, INTER_LANCZOS4); // 이미지 리사이징
                    cvtColor(colorMat, colorMat, COLOR_RGB2GRAY);
                    //cout << writer.isOpened() << '\n';

                    writer.write(colorMat);

                    // 저장된 영상 데이터가 올바른지 확인하기 위해 영상 데이터 크기를 출력합니다.
                    //std::streampos fileSize = writer.getStream().tellp();
                    //std::cout << "Frame " << frame_count << ": Saved " << fileSize / 1024 << "KB" << std::endl;
                    //cv::imshow("image", colorMat);

                    // 종료 키 대기
                    /*if ((key_input & 0xFF) == 27) break;*/ // esc로 종료

                /*  cap.release();*/
                }
                // num bodies check 조건
                k4a_image_release(depth);
                k4a_image_release(color_image);
                k4abt_frame_release(body_frame); // Remember to release the body frame once you finish using it
                /*
                k4abt_frame_release(body_frame2);
                k4a_image_release(depth2);
                k4a_image_release(color_image2);
                */
                // Remember to release the body frame once you finish using it

            }// body tracking check 조건(depth, color_image)

        }// capture check 조건 (body frame)
        double loops5 = clock();
        cout << "loop" << loops5 - loops << '\n';

    } while (frame_count < 30); // frame이 증가하는 루프


    writer.release();//7.27
    destroyAllWindows();//7.27
    printf("Finished body tracking processing!\n");

    k4abt_tracker_shutdown(tracker);
    k4abt_tracker_destroy(tracker);

    k4a_device_stop_cameras(device);
    k4a_device_close(device);

    //k4abt_tracker_shutdown(tracker2);
    //k4abt_tracker_destroy(tracker2);

    //k4a_device_stop_cameras(device2);
    //k4a_device_close(device2);



    return 0;
} // main 종료

void Print2p(float x, float y) {
    cout << " Print out 2 points, x : " << x << " y : " << y << '\n';
    return;
}
void Print3p(float x, float y, float z) {
    cout << "Print out 3 points , x : " << x << " y : " << y << " z " << z << '\n';
    return;
}
void vector_print(vector<vector<vector<float>>>& v) {
    //0,2,8,15, 27, 21,25 --> end of joints connection or joint has over 3 brances 
    for (int i = 0; i <= 27; i++) {
        if (i == 9 || i == 16 || i == 17 || i == 10 || i == 0 || i == 2 || i == 8 || i == 15 || i == 27 || i == 21 || i == 25) { continue; }
        cout << "Parent vector , index " << i << " x : " << v[i][0][0] << " y : " << v[i][0][1] << " z : " << v[i][0][2] << '\n';
        cout << "Child vector , index " << i << " x : " << v[i][1][0] << " y : " << v[i][1][1] << " z : " << v[i][1][2] << '\n';
    }
}
void CheckIntrinsicParam(k4a_calibration_t sensor_calibration) {
    vector<float> intrinsics;
    Mat cameraMatrix;
    intrinsics.assign(15, 0);
    for (int i = 0; i < 15; i++) {
        intrinsics[i] = sensor_calibration.depth_camera_calibration.intrinsics.parameters.v[i];
        cout << intrinsicArr[i] << " value : " << intrinsics[i] << '\n';
    }
    cout << '\n';
    /*cout << "cameraMatrix = " << endl << " " << cameraMatrix << endl << '\n';*/
}
void Make_vector(int index, Mat& vectors, Mat& p_set, int i_body) {
    /*cout << "index : " << index << " parnet index : " << joints_hierarchy[index] << '\n';*/
    float x_dif = p_set.at<float>(28 * i_body + index, 0) - p_set.at<float>(28 * i_body + joints_hierarchy[index], 0);
    float y_dif = p_set.at<float>(28 * i_body + index, 1) - p_set.at<float>(28 * i_body + joints_hierarchy[index], 1);
    float z_dif = p_set.at<float>(28 * i_body + index, 2) - p_set.at<float>(28 * i_body + joints_hierarchy[index], 2);
    float div = sqrt(pow(x_dif, 2) + pow(y_dif, 2) + pow(z_dif, 2));
    cout << "joint number : " << index << " " << " 3D vector info: " << x_dif << ", " << y_dif << ", " << z_dif << '\n';
    //vectors.at<float>(28 * i_body + index, 0) = x_dif / div;
    //vectors.at<float>(28 * i_body + index, 1) = y_dif / div;
    //vectors.at<float>(28 * i_body + index, 2) = z_dif / div;
}

void Make_vector2D(int index, Mat& p_set, int i_body) {
    /*cout << "index : " << index << " parnet index : " << joints_hierarchy[index] << '\n';*/
    float x_dif = p_set.at<float>(28 * i_body + index, 0) - p_set.at<float>(28 * i_body + joints_hierarchy[index], 0);
    float y_dif = p_set.at<float>(28 * i_body + index, 1) - p_set.at<float>(28 * i_body + joints_hierarchy[index], 1);
    float div = sqrt(pow(x_dif, 2) + pow(y_dif, 2));
    cout << "joint number : " << index << " " << " 2D vector info : " << x_dif << ", " << y_dif << " " << '\n';

}

void Get_theta(int index, Mat& vectors, float theta[], int i_body) {
    Mat c1 = vectors.row(28 * i_body + joints_hierarchy[index]);
    Mat c2 = vectors.row(28 * i_body + index);


    float cos = c1.dot(c2);
    float rad = acos(cos);

    // cout << "index : " << index << " theta  : " << res << '\n';

    theta[28 * i_body + index] = rad;
}
void Get_n_vector(int index, Mat& vectors, Mat& nv_set, int i_body) {
    Mat c1 = vectors.row(28 * i_body + joints_hierarchy[index]);
    Mat c2 = vectors.row(28 * i_body + index);

    Mat n_vector = c1.cross(c2);
    //cout << "cross : " << n_vector << '\n';
    nv_set.at<float>(28 * i_body + index, 0) = n_vector.at<float>(0);
    nv_set.at<float>(28 * i_body + index, 1) = n_vector.at<float>(1);
    nv_set.at<float>(28 * i_body + index, 2) = n_vector.at<float>(2);
}
void Get_rotation_matrix(int index, Mat& nv_set, float th_set[], Mat& rot_set, Mat& inv_rot_set, int i_body) {
    Rodrigues(nv_set.row(28 * i_body + index) * th_set[28 * i_body + index], rot_set);
    inv_rot_set = rot_set.inv();
    //cout << "in function rotation mat :" << rot_set << '\n';
}

void Print_joint_error(Mat pset1, Mat pset2, int i_body, string c) {
    string file_name;
    if (c == "12") {
        file_name = "joint_error12.csv";
        cout << "Print joint error kinect1->kinect2" << '\n';
    }
    else if (c == "21") {
        file_name = "joint_error21.csv";
        cout << "Print joint error kinect2->kinect1" << '\n';
    }

    std::ofstream ofs(file_name);
    double error_tot = 0;
    double x_tot = 0;
    double y_tot = 0;
    double z_tot = 0;
    for (int i = 0; i <= 27; i++) {
        if (i == 9 || i == 16 || i == 17 || i == 10) { continue; }
        double x_joint_error = abs(pset1.at<float>(28 * i_body + i, 0) - pset2.at<float>(28 * i_body + i, 0));
        double y_joint_error = abs(pset1.at<float>(28 * i_body + i, 1) - pset2.at<float>(28 * i_body + i, 1));
        double z_joint_error = abs(pset1.at<float>(28 * i_body + i, 2) - pset2.at<float>(28 * i_body + i, 2));
        double joint_error = sqrt(pow(x_joint_error, 2) + pow(y_joint_error, 2) + pow(z_joint_error, 2));
        cout << "index : " << i << " " << " joint error :" << joint_error << " x error :" << x_joint_error << " y error : " << y_joint_error << " z error : " << z_joint_error << '\n';
        ofs << "index," << i << "," << " joint error ," << joint_error << ", x error ," << x_joint_error << ", y error , " << y_joint_error << " z error , " << z_joint_error << '\n';
        error_tot += joint_error;
        x_tot += x_joint_error; y_tot += y_joint_error; z_tot += z_joint_error;
    }
    cout << "mean error : " << error_tot / 24 << " x mean error : " << x_tot / 24 << " y mean error : " << y_tot / 24 << " z mean error :  " << z_tot / 24 << '\n';
    ofs << "mean error : ," << error_tot / 24 << ", x mean error : ," << x_tot / 24 << ", y mean error : ," << y_tot / 24 << ", z mean error :  ," << z_tot / 24 << '\n';

}


void Multiply_extrinsic_matrix(Mat& pset1, Mat& pset2, Mat transform, int i_body) {
    for (int i = 0; i <= 27; i++) {
        if (i == 9 || i == 16 || i == 17 || i == 10) { continue; }
        Mat newpset1 = Mat::ones(4, 1, CV_32FC1);
        Mat newpset2 = Mat::ones(4, 1, CV_32FC1);
        newpset1.at<float>(0, 0) = pset1.at<float>(28 * i_body + i, 0);
        newpset1.at<float>(1, 0) = pset1.at<float>(28 * i_body + i, 1);
        newpset1.at<float>(2, 0) = pset1.at<float>(28 * i_body + i, 2);
        newpset2 = transform * newpset1;
        pset2.at<float>(28 * i_body + i, 0) = newpset2.at<float>(0, 0);
        pset2.at<float>(28 * i_body + i, 1) = newpset2.at<float>(1, 0);
        pset2.at<float>(28 * i_body + i, 2) = newpset2.at<float>(2, 0);
        //cout << "test extrinsic result: " << newpset2.at<float>(0,0) << " " << newpset2.at<float>(1,0) << " " << newpset2.at<float>(2,0) << '\n';
    }
}

void color() {
    int i = 0;
    Scalar blue{ 255,0,0 }; Scalar orange{ 0,128,255 }; Scalar yellow{ 0,255,255 }; Scalar green{ 0,255,128 }; Scalar red{ 0,0,255 }; Scalar gray{ 160,160,160 }; Scalar sky_blue{ 76,153,0 }; Scalar bright_gray{ 160,160,160 };

    colors[0] = bright_gray; // 중심
    colors[1] = gray; // 척추1
    colors[2] = gray; // 척추2
    colors[11] = red; colors[4] = red; //쇄골    
    colors[12] = orange; colors[5] = orange; //어깨
    colors[13] = sky_blue; colors[6] = sky_blue; // 팔꿈치
    colors[14] = Scalar(128, 255, 0); colors[7] = Scalar(128, 255, 0); //손목
    colors[15] = Scalar(0, 153, 76); colors[8] = Scalar(0, 153, 76); //손
    colors[3] = Scalar(102, 0, 204); //목
    colors[26] = Scalar(255, 0, 255); //얼굴
    colors[27] = Scalar(255, 153, 255); //코
    colors[22] = red; colors[23] = orange; colors[24] = yellow; colors[25] = green;//우측하체
    colors[18] = red; colors[19] = orange; colors[20] = yellow; colors[21] = green;//좌측하체
}

void color0() {
    int i;
    for (i = 0; i < 28; i++) {
        colors0[i] = Scalar(255, 255, 255);
    }

}

static void Get_3dPoints(vector<k4abt_skeleton_t> skeleton, int i_body, Mat& p_set) {
    for (int i = 0; i <= 27; i++) {
        if (i == 9 || i == 16 || i == 17 || i == 10) { continue; }
        int num_j = i;
        float x = skeleton[i_body].joints[num_j].position.xyz.x;
        float y = skeleton[i_body].joints[num_j].position.xyz.y;
        float z = skeleton[i_body].joints[num_j].position.xyz.z; // 24*3 
        p_set.at<float>(28 * i_body + i, 0) = x;
        p_set.at<float>(28 * i_body + i, 1) = y;
        p_set.at<float>(28 * i_body + i, 2) = z;
    }
}

static void  Drawing_circle(vector<k4abt_skeleton_t> skeleton, int i_body, Mat& p_set, k4a_calibration_t sensor_calibration, k4a_float2_t xy_color[], Mat& p_set_2D, k4abt_frame_t body_frame, Mat& colorMat,
    k4a_image_t color_image, uint8_t* color_buffer, Scalar colors[]) {
    for (int i = 0; i <= 27; i++) {
        if (i == 9 || i == 16 || i == 17 || i == 10) { continue; }
        int num_j = i;
        int re = 0;
        k4a_float3_t points3d;
        points3d.xyz.x = p_set.at<float>(28 * i_body + i, 0);
        points3d.xyz.y = p_set.at<float>(28 * i_body + i, 1);
        points3d.xyz.z = p_set.at<float>(28 * i_body + i, 2);

        //cout << p_set.row(i) << '\n';
        k4a_calibration_3d_to_2d(&sensor_calibration, &points3d, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR, &xy_color[28 * i_body + i], &re);
        p_set_2D.at<float>(28 * i_body + i, 0) = xy_color[28 * i_body + i].xy.x;
        p_set_2D.at<float>(28 * i_body + i, 1) = xy_color[28 * i_body + i].xy.y;
        uint32_t id = k4abt_frame_get_body_id(body_frame, i);
        if (color_image != NULL) {
            if (color_buffer != NULL) {
                cv::circle(colorMat, Point(xy_color[28 * i_body + i].xy.x, xy_color[28 * i_body + i].xy.y), 7, colors[i], 4, 1, 0);
            }
        }
    }
}
static void Drawing_marker(vector<k4abt_skeleton_t> skeleton, int i_body, Mat& p_set, k4a_calibration_t sensor_calibration, k4a_float2_t xy_color[], Mat& p_set_2D, k4abt_frame_t body_frame, Mat& colorMat,
    k4a_image_t color_image, uint8_t* color_buffer, Scalar colors[]) {
    for (int i = 0; i <= 27; i++) {
        if (i == 9 || i == 16 || i == 17 || i == 10) { continue; }
        int num_j = i;
        int re = 0;
        k4a_float3_t points3d;
        points3d.xyz.x = p_set.at<float>(28 * i_body + i, 0);
        points3d.xyz.y = p_set.at<float>(28 * i_body + i, 1);
        points3d.xyz.z = p_set.at<float>(28 * i_body + i, 2);

        //cout << p_set.row(i) << '\n';
        k4a_calibration_3d_to_2d(&sensor_calibration, &points3d, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR, &xy_color[28 * i_body + i], &re);
        p_set_2D.at<float>(28 * i_body + i, 0) = xy_color[28 * i_body + i].xy.x;
        p_set_2D.at<float>(28 * i_body + i, 1) = xy_color[28 * i_body + i].xy.y;
        uint32_t id = k4abt_frame_get_body_id(body_frame, i);
        if (color_image != NULL) {
            if (color_buffer != NULL) {
                cv::drawMarker(colorMat, Point(xy_color[28 * i_body + i].xy.x, xy_color[28 * i_body + i].xy.y), colors[i], 0, 10, 5, 0);
            }
        }
    }
}


void static Comparison_Rod(int num_bodies, Mat rod_l[]) {
    if (num_bodies != 2) {
        cout << "compare error : numbody \n";
        return;
    }
    float norm1_sum;
    Mat R1, R2;
    float diff_cossim = 0;
    float diff_rotation = 0;
    for (int j = 0; j <= 27; j++) {
        if (j == 9 || j == 16 || j == 17 || j == 10 || j == 0) { continue; }
        Rodrigues(rod_l[j], R1);
        Rodrigues(rod_l[28 * (num_bodies - 1) + j], R2);

        diff_cossim += abs((rod_l[j].dot(rod_l[28 * (num_bodies - 1) + j])) / (norm(rod_l[j], cv::NORM_L2) * norm(rod_l[28 * (num_bodies - 1) + j], cv::NORM_L2))); // cosine similarity
        /*cout << " joint " << j << " cossim diff : " << abs((rod_l[j].dot(rod_l[28 * (num_bodies - 1) + j])) / (norm(rod_l[j], cv::NORM_L2) * norm(rod_l[28 * (num_bodies - 1) + j], cv::NORM_L2))) << '\n';*/

        diff_rotation += norm(R1 - R2); // rotation similarity
        /* cout << " joint " << j << " rotation diff : "  << norm(R1 - R2) << "\n";   */

    }
    cout << "diff by cossim " << diff_cossim << "\n";
    cout << "diff by rotation: " << diff_rotation << "\n";
}
void static create_xy_table(const k4a_calibration_t* calibration, k4a_image_t xy_table)
{
    k4a_float2_t* table_data = (k4a_float2_t*)(void*)k4a_image_get_buffer(xy_table);

    int width = calibration->depth_camera_calibration.resolution_width;
    int height = calibration->depth_camera_calibration.resolution_height;

    k4a_float2_t p;
    k4a_float3_t ray;
    int valid;

    for (int y = 0, idx = 0; y < height; y++)
    {
        p.xy.y = (float)y;
        for (int x = 0; x < width; x++, idx++)
        {
            p.xy.x = (float)x;

            k4a_calibration_2d_to_3d(
                calibration, &p, 1.f, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_DEPTH, &ray, &valid);

            if (valid)
            {
                table_data[idx].xy.x = ray.xyz.x;
                table_data[idx].xy.y = ray.xyz.y;
            }
            else
            {
                table_data[idx].xy.x = nanf("");
                table_data[idx].xy.y = nanf("");
            }
        }
    }
}



