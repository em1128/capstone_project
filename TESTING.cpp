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

#define VERIFY(result, error)                                                                            \
    if(result != K4A_RESULT_SUCCEEDED)                                                                   \
    {                                                                                                    \
        printf("%s \n - (File: %s, Function: %s, Line: %d)\n", error, __FILE__, __FUNCTION__, __LINE__); \
        exit(1);                                                                                         \
    }                                                                                                    \


int altered_index(int index) {
    switch (index) {
    case 0 :
        return 0; //Hips
        break;
    case 18 :
        return 1; //Leftuplegs
        break;
    case 22:
        return 2;
        break;
    case 19:
        return 4;
        break;
    case 23:
        return 5;
        break;
    case 1:
        return 6;
        break;
    case 20:
        return 7;
        break;
    case 24:
        return 8;
        break;
    case 2:
        return 9;
        break;
    case 21:
        return 10;
        break;
    case 25:
        return 11;
        break;
    case 3:
        return 12;
        break;
    case 4:
        return 13;
        break;
    case 11:
        return 14;
        break;
    case 27:
        return 19;
        break;
    case 5:
        return 16;
        break;
    case 12:
        return 17;
        break;
    case 6:
        return 18;
        break;
    case 13:
        return 19;
        break;
    case 7:
        return 20;
        break;
    case 14:
        return 21;
        break;
    case 8:
        return 22;
        break;
    case 15:
        return 23;
        break;
    case 26:
        return 24;
        break;
    }
}

int main()
{
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

    /*vector<float> intrinsics;
    intrinsics.assign(15, 0);
    for (int i = 0; i < 15; i++) {
        intrinsics[i] = sensor_calibration.depth_camera_calibration.intrinsics.parameters.v[i];
        cout << "index : " << i << " value : " << intrinsics[i] << '\n';
    }*/


    k4abt_tracker_t tracker = NULL;
    k4abt_tracker_configuration_t tracker_config = K4ABT_TRACKER_CONFIG_DEFAULT;
    VERIFY(k4abt_tracker_create(&sensor_calibration, tracker_config, &tracker), "Body tracker initialization failed!");

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
                
                printf("%zu bodies are detected!\n", num_bodies);
                for (size_t i = 0; i < num_bodies; i++)
                {
                    k4abt_skeleton_t skeleton;
                    k4abt_frame_get_body_skeleton(body_frame, i, &skeleton);
                    uint8_t* color_buffer = k4a_image_get_buffer(color_image);
                    cv::namedWindow("image"); // 이미지를 보여주기 위한 빈 창
                    cv::Mat colorMat(720, 1280, CV_8UC4, (void*)color_buffer, cv::Mat::AUTO_STEP);
                    for (int i = 0; i <= 27; i++) {
                        if (i == 9 || i == 16 || i == 17 || i == 10) { continue; }
                        int num_j = i;
                        float x = skeleton.joints[num_j].position.xyz.x;
                        float y = skeleton.joints[num_j].position.xyz.y;
                        float z = skeleton.joints[num_j].position.xyz.z; // 24*3 
                        k4a_float2_t xy_color;
                        int re = 0;
                        k4a_calibration_3d_to_2d(&sensor_calibration, &skeleton.joints[num_j].position, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR, &xy_color, &re);

                        cout << "x :" << skeleton.joints[8].position.xyz.x << " y : " << skeleton.joints[8].position.xyz.y << " z : " << skeleton.joints[8].position.xyz.z <<'\n';
                        cout << "using calbration function x : " << xy_color.xy.x << " y : " << xy_color.xy.y << '\n';
                        uint32_t id = k4abt_frame_get_body_id(body_frame, i);
                        if (color_image != NULL) {                          
                            if (color_buffer != NULL) {                              
                                
                                cv::circle(colorMat, Point(xy_color.xy.x, xy_color.xy.y), 7, Scalar(255, 0, 0), 4, 1, 0);
                                
                            }
                        }

                    }
                    cv::imshow("image", colorMat); // "image"라는 이름의 창에 이미지를 넣어 보여줌

                    cv::waitKey(); // 종료 키 대기
                    cout << '\n';
                }
               
                2 ^ 3;
                


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