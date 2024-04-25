Azure kinect DK 필요(2대 활용)
https://learn.microsoft.com/ko-kr/azure/kinect-dk/set-up-azure-kinect-dk

Visual Studio에서 다음 package들을 설치해야 함.
- Microsoft.Azure.Kinect.Sensor.1.4.1
- Microsoft.Azure.Kinect.BodyTracking.ONNXRuntime.1.10.0
- Microsoft.Azure.Kinect.BodyTracking.Dependencies.cuDNN.0.9.1
- Microsoft.Azure.Kinect.BodyTracking.Dependencies.0.9.1
- Microsoft.Azure.Kinect.BodyTracking.1.1.2

dnn_model_2_0_op11.onnx

required opencv-4.7.0 
opencv_world470.dll이 project 내에 존재해야 함.
opencv_world470d.dll이 project 내에 존재해야 함.

--- 

Kinect로 사람을 관측해서 관절의 좌표를 구하고, 
각 관절을 시작점으로 하는 회전각으로 구해서 자세의 유사도를 비교.
이후 모범 사례와 비교하여 점수를 측정하는 시스템.

Kinect 2대를 이용해서 각각의 카메라에서 얻은 관절에 신뢰도를 적용하여 정확도를 높임.
회전각은 rodrigues vector를 이용하여 구함.
관절마다의 local 좌표계로 변환해서 관절별로 비교.
cosine 유사도를 활용하여 자세의 유사도를 측정.
DTW를 활용하여 영상별 다른 동작시간에 의한 차이를 줄임.

https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11742704
