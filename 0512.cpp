#include <stdio.h>
#include <stdlib.h>


#include <opencv2/opencv.hpp>
#include <k4abttypes.h>
#include <k4a/k4a.h>
#include <k4abt.h>
#include <iostream>
#include<vector>

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
//                           //                
//                          23               19
//                          II               II
//                          24               20
//                          II               II
//                          25               21

static vector<int> joints_hierarchy;
static int vsize[2] = { 56, 3 }; // vector size
static  int rsize[3] = { 56, 3 ,3 }; //matrix size
//  ߰  0407
static  int vsize_2D[2] = { 56,2 }; // vector 2D size
static  Scalar colors[28];

void  static Hierarchy_set();

//check 2points
void static  Print2p(float x, float y);
//check 3points
void static  Print3p(float x, float y, float z);
//check vector
void static vector_print(vector<vector<vector<float>>>& v);

void static CheckIntrinsicParam(k4a_calibration_t sensor_calibration);

void static Make_vector(int index, Mat& vectors, Mat& p_set, int i_body);

void static Get_theta(int index, Mat& vectors, float theta[], int i_body);
void static Get_n_vector(int index, Mat& vectors, Mat& nv_set, int i_body);
void static Get_rotation_matrix(int index, Mat& nv_set, float th_set[], Mat& rot_set, Mat& inv_rot_set, int i_body);

void static color() {
    int i = 0;
    for (i = 0; i < 28; i++) {
        if (i % 4 == 0) {
            colors[i] = Scalar(80 + 8 * i, 80, 80 + 8 * i);
        }
        else if (i % 4 == 1) {
            colors[i] = Scalar(80, 80 + 8 * i, 80 + 8 * i);
        }
        else if (i % 4 == 2) {
            colors[i] = Scalar(80, 80, 80 + 8 * i);
        }
        else if (i % 4 == 3) {
            colors[i] = Scalar(80, 80 + 8 * i, 80);
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

        diff_cossim += 1 - abs((rod_l[j].dot(rod_l[28 * (num_bodies - 1) + j])) / (norm(rod_l[j], cv::NORM_L2) * norm(rod_l[28 * (num_bodies - 1) + j], cv::NORM_L2))); // cosine similarity
        /*cout << " joint " << j << " cossim diff : " << abs((rod_l[j].dot(rod_l[28 * (num_bodies - 1) + j])) / (norm(rod_l[j], cv::NORM_L2) * norm(rod_l[28 * (num_bodies - 1) + j], cv::NORM_L2))) << '\n';*/

        diff_rotation += norm(R1 - R2); // rotation similarity
        /* cout << " joint " << j << " rotation diff : "  << norm(R1 - R2) << "\n";   */

    }
    cout << "diff by cossim " << diff_cossim << "\n";
    cout << "diff by rotation: " << diff_rotation << "\n";
}

int main()
{
    //Hierarchy_initalize
    Hierarchy_set();
    color();
    k4a_device_t device = NULL;
    VERIFY(k4a_device_open(0, &device), "Open K4A Device failed!");

    // Start camera. Make sure depth camera is enabled.
    k4a_device_configuration_t deviceConfig = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    deviceConfig.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    deviceConfig.camera_fps = K4A_FRAMES_PER_SECOND_5;
    deviceConfig.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
    deviceConfig.color_resolution = K4A_COLOR_RESOLUTION_720P;
    VERIFY(k4a_device_start_cameras(device, &deviceConfig), "Start K4A cameras failed!");

    k4a_calibration_t sensor_calibration;
    VERIFY(k4a_device_get_calibration(device, deviceConfig.depth_mode, deviceConfig.color_resolution, &sensor_calibration),
        "Get depth camera calibration failed!");

    //Check intrinsics info
    CheckIntrinsicParam(sensor_calibration);

    //traker config
    k4abt_tracker_t tracker = NULL;
    k4abt_tracker_configuration_t tracker_config = K4ABT_TRACKER_CONFIG_DEFAULT;
    VERIFY(k4abt_tracker_create(&sensor_calibration, tracker_config, &tracker), "Body tracker initialization failed!");

    //frame start
    int frame_count = 0;
    do
    {
        k4a_capture_t sensor_capture;
        k4a_wait_result_t get_capture_result = k4a_device_get_capture(device, &sensor_capture, K4A_WAIT_INFINITE);
        if (get_capture_result == K4A_WAIT_RESULT_SUCCEEDED)
        {
            frame_count++;
            k4a_wait_result_t queue_capture_result = k4abt_tracker_enqueue_capture(tracker, sensor_capture, K4A_WAIT_INFINITE);
            k4a_capture_release(sensor_capture); // Remember to release the sensor capture once you finish using it
            if (queue_capture_result == K4A_WAIT_RESULT_TIMEOUT)
            {
                // It should never hit timeout when K4A_WAIT_INFINITE is set.
                printf("Error! Add capture to tracker process queue timeout!\n");
                break;
            }
            else if (queue_capture_result == K4A_WAIT_RESULT_FAILED)
            {
                printf("Error! Add capture to tracker process queue failed!\n");
                break;
            }

            k4abt_frame_t body_frame = NULL;
            k4a_wait_result_t pop_frame_result = k4abt_tracker_pop_result(tracker, &body_frame, K4A_WAIT_INFINITE);

            if (pop_frame_result == K4A_WAIT_RESULT_SUCCEEDED)
            {
                // Successfully popped the body tracking result. Start your processing
                k4a_image_t depth = k4a_capture_get_depth_image(sensor_capture);
                k4a_image_t color_image = k4a_capture_get_color_image(sensor_capture);
                size_t num_bodies = k4abt_frame_get_num_bodies(body_frame);


                //initalize position & store vecotr of parnet, child
                //::Mat joints_vector(24, 2, 3);
                Mat v_set(2, vsize, CV_32FC1, Scalar(0)); // vectors set 28*3
                v_set.at<float>(0, 0) = 0; //false pelvis vector set
                v_set.at<float>(0, 1) = -1;
                v_set.at<float>(0, 2) = 0;

                v_set.at<float>(28, 0) = 0; //false pelvis vector set -->        
                v_set.at<float>(28, 1) = -1;
                v_set.at<float>(28, 2) = 0;
                // 0407  ߰ 
                Mat p_set_2D(2, vsize_2D, CV_32FC1, Scalar(0)); // vectors set 28*2

                Mat p_set(2, vsize, CV_32FC1, Scalar(0)); //3d points set 28*3
                float th_set[56] = { 0, }; //theta set 28*1 // 0407      27->28
                Mat rot_set[56]; // rotation matrix set 28 * 3 * 3
                Mat inv_rot_set[56]; //inverse rotation matrix set 28 * 3 * 3
                Mat nv_set(2, vsize, CV_32FC1, Scalar(0)); //n_vector set 28*3
                Mat frames[56]; //  ǥ   28  
                Mat rod_l[56]; // ε帮 Խ       local
                Mat rod_g[56]; // ε帮 Խ       global



                /* printf("%zu bodies are detected!\n", num_bodies);*/
                if (num_bodies) {
                    uint8_t* color_buffer = k4a_image_get_buffer(color_image);
                    cv::namedWindow("image"); //  ̹           ֱ          â
                    cv::Mat colorMat(720, 1280, CV_8UC4, (void*)color_buffer, cv::Mat::AUTO_STEP); // color camera  ̹   
                    cout << "        ? :" << num_bodies << '\n';
                    vector<k4abt_skeleton_t> skeleton(num_bodies);
                    k4a_float2_t xy_color[56];

                    for (size_t i_body = 0; i_body < num_bodies; i_body++)
                    {

                        k4abt_frame_get_body_skeleton(body_frame, i_body, &skeleton[i_body]);
                        // 0407  ߰ 
                        cv::Mat cpColorMat(720, 1280, CV_8UC4); // copy  ̹   
                        colorMat.copyTo(cpColorMat);
                        cv::Mat grayMat(720, 1280, CV_8UC1); // gray  ̹   
                        cv::cvtColor(colorMat, grayMat, COLOR_BGR2GRAY);
                        //


                        for (int i = 0; i <= 27; i++) {
                            if (i == 9 || i == 16 || i == 17 || i == 10) { continue; }
                            int num_j = i;
                            float x = skeleton[i_body].joints[num_j].position.xyz.x;
                            float y = skeleton[i_body].joints[num_j].position.xyz.y;
                            float z = skeleton[i_body].joints[num_j].position.xyz.z; // 24*3 
                            p_set.at<float>(28 * i_body + i, 0) = x;
                            p_set.at<float>(28 * i_body + i, 1) = y;
                            p_set.at<float>(28 * i_body + i, 2) = z;

                            int re = 0;
                            k4a_calibration_3d_to_2d(&sensor_calibration, &skeleton[i_body].joints[num_j].position, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR, &xy_color[28 * i_body + i], &re);
                            //0407  ߰ 
                            p_set_2D.at<float>(28 * i_body + i, 0) = xy_color[28 * i_body + i].xy.x;
                            p_set_2D.at<float>(28 * i_body + i, 1) = xy_color[28 * i_body + i].xy.y;
                            //

                            //Print3p(skeleton.joints[8].position.xyz.x, skeleton.joints[8].position.xyz.y, skeleton.joints[8].position.xyz.z);
                            //Print2p(xy_color.xy.x, xy_color.xy.y);
                            uint32_t id = k4abt_frame_get_body_id(body_frame, i);
                            // joint xyz print
                            //cout << i <<"xyz is x:" << skeleton.joints[i].position.xyz.x << "y:" << skeleton.joints[i].position.xyz.y << "z:" << skeleton.joints[i].position.xyz.z << '\n';                   
                            if (color_image != NULL) {
                                if (color_buffer != NULL) {
                                    cv::circle(colorMat, Point(xy_color[28 * i_body + i].xy.x, xy_color[28 * i_body + i].xy.y), 7, Scalar(255, 0, 0), 4, 1, 0);
                                }
                            }

                        }
                        // 0407  ߰ 
                        // save data
                        cv::FileStorage fs("saveMat.xml", cv::FileStorage::WRITE);

                        fs << "joint3D" << p_set;
                        fs << "joint2D" << p_set_2D;
                        fs << "grayImage" << grayMat;
                        fs.release();
                        // 

                        //drawing skeleton on image 
                        for (int j = 0; j <= 27; j++) {
                            if (j == 9 || j == 16 || j == 17 || j == 10 || j == 0) { continue; }
                            cv::line(colorMat, Point(xy_color[28 * i_body + j].xy.x, xy_color[28 * i_body + j].xy.y), Point(xy_color[28 * i_body + joints_hierarchy[j]].xy.x, xy_color[28 * i_body + joints_hierarchy[j]].xy.y), colors[j], 3, 1, 0);
                        }

                        for (int j = 0; j <= 27; j++) {
                            if (j == 9 || j == 16 || j == 17 || j == 10 || j == 0) { continue; }
                            Make_vector(j, v_set, p_set, i_body); // ε       θ    ǥ      ε      ǥ        3         
                            //cout << "index " << j << v_set.at<float>(j, 0) << "  " << v_set.at<float>(j, 1) << " " << v_set.at<float>(j, 2) << '\n';
                        }
                        for (int j = 0; j <= 27; j++) {
                            if (j == 9 || j == 16 || j == 17 || j == 10 || j == 0) { continue; }
                            Get_theta(j, v_set, th_set, i_body);  // ε       θ ,  ε       θ     θ       ؼ          ,radian            
                            Get_n_vector(j, v_set, nv_set, i_body); // ε帮 Խ            
                            Get_rotation_matrix(j, nv_set, th_set, rot_set[28 * i_body + j], inv_rot_set[28 * i_body + j], i_body); //     ̼    Ʈ        ϱ , inv      ̼    Ʈ        ϱ 
                            cout << '\n';
                        }


                        //global  ε帮 Խ            
                        for (int j = 0; j <= 27; j++) {
                            if (j == 9 || j == 16 || j == 17 || j == 10 || j == 0) { continue; }
                            rod_g[28 * i_body + j] = th_set[28 * i_body + j] * nv_set.row(28 * i_body + j);
                        }
                        frames[0] = Mat::eye(3, 3, CV_32F);
                        frames[28] = Mat::eye(3, 3, CV_32F);
                        //make local rodrigues vector           
                        for (int i = 0; i <= 27; i++) {
                            if (i == 9 || i == 16 || i == 17 || i == 10 || i == 0) { continue; }
                            rod_l[28 * i_body + i] = (frames[28 * i_body + joints_hierarchy[i]].inv() * rod_g[28 * i_body + i].t()).t();
                            frames[28 * i_body + i] = rot_set[28 * i_body + i] * frames[28 * i_body + joints_hierarchy[i]];
                        }

                        //showing joint vector  info
                        cout << "vectors of joints " << '\n';
                        for (int j = 0; j <= 27; j++) {
                            if (j == 9 || j == 16 || j == 17 || j == 10) { continue; }
                            cout << "index  " << j << " " << v_set.row(j) << '\n';
                        }
                        //showing global rod_vector  info
                        cout << '\n' << "global rodrigues vectors of joints " << '\n';
                        for (int j = 0; j <= 27; j++) {
                            if (j == 9 || j == 16 || j == 17 || j == 10 || j == 0) { continue; }
                            cout << "index  " << j << " " << rod_g[j] << '\n';
                        }
                        //showing local rod_vector  info
                        cout << '\n' << "local rodrigues vectors of joints " << '\n';
                        for (int j = 0; j <= 27; j++) {
                            if (j == 9 || j == 16 || j == 17 || j == 10 || j == 0) { continue; }
                            cout << "index  " << j << " " << rod_l[j] << '\n';
                        }


                        //k4abt_joint_confidence_level_t confidence_level[28];
                        //for (int i = 0; i < 28; i++) {
                        //    confidence_level[28 * i_body + i] = skeleton[i_body].joints[i].confidence_level;
                        //    if (skeleton[i_body].joints[i].confidence_level == K4ABT_JOINT_CONFIDENCE_LOW) {
                        //        cout << "confidence level of joint" << i << ": " << 0 << '\n';
                        //    }
                        //    else if (skeleton[i_body].joints[i].confidence_level == K4ABT_JOINT_CONFIDENCE_MEDIUM) {
                        //        cout << "confidence level of joint" << i << ": " << 1 << '\n';
                        //    }
                        //    else if (skeleton[i_body].joints[i].confidence_level == K4ABT_JOINT_CONFIDENCE_HIGH) {
                        //        cout << "confidence level of joint" << i << ": " << 2 << '\n';
                        //    }
                        //    else if (skeleton[i_body].joints[i].confidence_level == K4ABT_JOINT_CONFIDENCE_NONE) {
                        //        cout << "confidence level of joint" << i << ": " << 3 << '\n';
                        //    }
                        //}


                    }
                    //    Լ      .
                    Comparison_Rod(num_bodies, rod_l);


                    cv::imshow("image", colorMat); // "image"     ̸    â    ̹       ־        
                    // 0407          ߰ 
                    int key_input = cv::waitKey(); //      Ű    
                    if ((key_input & 0xFF) == 27) break; // esc       
                    //
                    cout << '\n';
                }


                k4a_image_release(depth);
                k4a_image_release(color_image);
                k4abt_frame_release(body_frame); // Remember to release the body frame once you finish using it
            }
            else if (pop_frame_result == K4A_WAIT_RESULT_TIMEOUT)
            {
                //  It should never hit timeout when K4A_WAIT_INFINITE is set.
                printf("Error! Pop body frame result timeout!\n");
                break;
            }
            else
            {
                printf("Pop body frame result failed!\n");
                break;
            }
        }
        else if (get_capture_result == K4A_WAIT_RESULT_TIMEOUT)
        {
            // It should never hit time out when K4A_WAIT_INFINITE is set.
            printf("Error! Get depth frame time out!\n");
            break;
        }
        else
        {
            printf("Get depth capture returned error: %d\n", get_capture_result);
            break;
        }

    } while (frame_count < 100);

    printf("Finished body tracking processing!\n");

    k4abt_tracker_shutdown(tracker);
    k4abt_tracker_destroy(tracker);

    k4a_device_stop_cameras(device);
    k4a_device_close(device);

    return 0;
}

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
        cout << "index : " << i << " value : " << intrinsics[i] << '\n';
    }
    /*cout << "cameraMatrix = " << endl << " " << cameraMatrix << endl << '\n';*/
}
void Make_vector(int index, Mat& vectors, Mat& p_set, int i_body) {
    cout << "index : " << index << " parnet index : " << joints_hierarchy[index] << '\n';
    float x_dif = p_set.at<float>(28 * i_body + index, 0) - p_set.at<float>(28 * i_body + joints_hierarchy[index], 0);
    float y_dif = p_set.at<float>(28 * i_body + index, 1) - p_set.at<float>(28 * i_body + joints_hierarchy[index], 1);
    float z_dif = p_set.at<float>(28 * i_body + index, 2) - p_set.at<float>(28 * i_body + joints_hierarchy[index], 2);
    float div = sqrt(pow(x_dif, 2) + pow(y_dif, 2) + pow(z_dif, 2));
    //cout << x_dif << " " << y_dif << " " << z_dif << '\n';
    vectors.at<float>(28 * i_body + index, 0) = x_dif / div;
    vectors.at<float>(28 * i_body + index, 1) = y_dif / div;
    vectors.at<float>(28 * i_body + index, 2) = z_dif / div;
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