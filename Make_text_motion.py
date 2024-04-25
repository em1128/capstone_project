from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2 as cv

#data2 = np.load('Taeguek01.npy')

#print(data2.shape) # 25 * 3 # 3094
# we are using 25 * 3 * ( 0 ~ 1900)
# 18 motion sequences

draw_text = ["준비서기(나란히서기)","내딛어 왼 앞서기 아래막기","내딛어 오른 앞서기 몸통 반대지르기","뒤로돌아 왼 앞서기 아래막기","내딛어 왼 앞서기 몸통 반대 지르기","돌아 왼 앞굽이 왼 아래막으며 왼 앞굽이 몸통 바로지르기"
                        ,"옮겨딛어 오른 앞서기 왼 몸통 안 막기","내딛어 왼 앞서기 몸통 바로지르기","뒤로돌아 왼 앞서기 오른 몸통안막기","내딛어 오른 앞서기 왼 몸통바로지르기","돌아 오른앞굽이 오른 아래막기 그대로 몸통바로지르기"
                        ,"옮겨딛어 왼 앞서기 왼 얼굴막기","오른 앞차고 오른 앞서기 몸통 반대지르기","뒤로돌아 오른 앞서기 오른 얼굴막기","왼 앞차고 왼 앞서기 몸통 반대지르기","옮겨딛어 왼 앞굽이 아래막기","내딛어 오른 앞굽이 몸통 반대지르기"
                        ,"바로(왼발 당겨오며 다시 준비자세)"]

font = ImageFont.truetype("C:/Windows/Fonts/gulim.ttc", 30)
#print(data2[0][0][0:2000])


print(len(draw_text))

for i in range(len(draw_text)):
    text_width = 30*len(draw_text[i])
    text_height = 30
    canvas = Image.new('RGB', (text_width, text_height), "white")
    draw = ImageDraw.Draw(canvas)
    print(draw_text[i])
    w, h = font.getsize(draw_text[i])
    draw.text(((text_width-w)/2.0,(text_height-h)/2.0), draw_text[i], 'black', font)
    canvas.save('./instruction_text/'+ str(i) +'_'+ draw_text[i]+'.png', "PNG")

print("done")





# back_ground = np.zeros((512,1024,3),np.uint8)
# def Show(a):
#     path  =   './instruction_text/'+'panorama'+'.png'
#     back_ground  = cv.imread(path)
#     path  =   './instruction_text/'+draw_text[a]+'.png'
#     img_array = np.fromfile(path, np.uint8) # 컴퓨터가 읽을수 있게 넘파이로 변환
#     ins_text = cv.imdecode(img_array,  cv.IMREAD_COLOR) #이미지를 읽어옴
#     back_ground[back_ground.shape[0]-ins_text.shape[0]:back_ground.shape[0],0:ins_text.shape[1]]= ins_text
#     cv.imshow('back_ground_with_text'+draw_text[i],back_ground)

# Show(18)
