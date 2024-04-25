# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 13:56:05 2023

@author: mingyu
"""

import re

# 파일을 읽어들일 텍스트 파일의 경로
file_path = 'test.txt'

try:
    # 파일 열기
    with open(file_path, 'r', encoding='utf-8') as file:
        # 파일 내용 읽기
        text= file.read()
        
except FileNotFoundError:
    print(f'{file_path} 파일을 찾을 수 없습니다.')
except Exception as e:
    print(f'파일 읽기 중 오류 발생: {e}')

# 정규 표현식을 사용하여 숫자를 추출
#11
pattern1_1 = r'kinect 1 confidecne level : 1\nkinect 2 confidecne level : 1\n유클리디안 거리 : (\d+\.\d+)'
matches1_1 = re.findall(pattern1_1, text)

#011
pattern01_1 = r'kinect 1 confidecne level : 0.1\nkinect 2 confidecne level : 1\n유클리디안 거리 : (\d+\.\d+)'
matches01_1 = re.findall(pattern01_1, text)

#101
pattern1_01 = r'kinect 1 confidecne level : 1\nkinect 2 confidecne level : 0.1\n유클리디안 거리 : (\d+\.\d+)'
matches1_01 = re.findall(pattern1_01, text)

#0011
pattern001_1 = r'kinect 1 confidecne level : 0.01\nkinect 2 confidecne level : 1\n유클리디안 거리 : (\d+\.\d+)'
matches001_1 = re.findall(pattern001_1, text)

#1001
pattern1_001 = r'kinect 1 confidecne level : 0.01\nkinect 2 confidecne level : 1\n유클리디안 거리 : (\d+\.\d+)'
matches1_001 = re.findall(pattern1_001, text)


# 문자열을 공백을 기준으로 분할하여 숫자로 변환한 후 리스트에 저장
numbers1 = [float(x) for x in matches01_1]
numbers2 = [float(y) for y in matches1_01]

# 숫자들의 평균을 계산
average = sum(numbers1+numbers2) / len(numbers1+numbers2)
# 평균 출력
print("평균:", average)
    
    
    
   