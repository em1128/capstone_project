import pandas as pd
import numpy as np
# npy 로드
original_array = np.load('D:\학교수업\캡디\참고\Taeguek01.npy')

n_joints, n_xyz, n_frames = original_array.shape

# 주어진 조인트 순서와 변경된 조인트 순서를 정의합니다.
# original_joint_order = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
fbx2kin_joint_order = [0, 18, 22, 'x',19, 23, 1, 20, 24, 2, 21, 25, 3, 4, 11, 27, 5, 12, 6, 13, 7, 14, 8, 15, 26]

# 조인트 순서를 변경하기 위해 새로운 배열을 초기화합니다.
n_kinect_joints = 28
kinect_array = np.zeros((n_kinect_joints,3, n_frames))

# 조인트 순서를 변경하면서 데이터를 복사합니다.
for i in range(n_joints):
    if i==3:
        continue
    new_joint_index = fbx2kin_joint_order[i]
    for j in range(3):
        x=np.zeros((3,n_frames))
        y=np.zeros((3,n_frames))
        z=np.zeros((3,n_frames))

        x,y,z = original_array[i, :, :]
        kinect_array[new_joint_index,: , :] = [x,-z,y]

# 배열 shape 변경
new_shape = (n_frames*n_kinect_joints, 3)
array_reshaped = np.transpose(kinect_array, (2, 0, 1)).reshape(new_shape)

# 동작별로 나눠서 csv 저장 
begin_frames = [0,367,440,510,580,640,720,780,850,940,1000,1080,1160,1260,1340,1440,1520,1580]
end_frames = [367,440,510,580,640,720,780,850,940,1000,1080,1160,1260,1340,1440,1520,1580,1900]

#total_data = np.zeros((len(end_frames),n_frames,n_kinect_joints,3))

for i in range(len(end_frames)):
    data = array_reshaped[begin_frames[i]*n_kinect_joints:end_frames[i]*28,:]
    
    #numpy array을 pandas dataframe으로 변환
    df = pd.DataFrame(data)
    file_name = 'D:\학교수업\캡디\코드\motion_'+str(i)+'.csv'
    df.to_csv(file_name, header=False, index=False,float_format = '%.7f')

    

    