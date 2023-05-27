import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO, YOLO_ONNX

import os
from tqdm import tqdm
import matplotlib.pyplot as plt

yolo = YOLO()

import easyocr
reader = easyocr.Reader(['ch_sim','en'])

def enhance_contrast(image, alpha, beta):
    # 对图像进行对比度增强
    enhanced_image = np.clip(alpha * np.array(image) + beta, 0, 255).astype(np.uint8)
    return enhanced_image
# 设置对比度增强的参数
alpha = 1.5  # 对比度增强因子
beta = 10 # 亮度增强因子

import re

def remove_special_characters(text):
    # 使用正则表达式匹配除了数字和英文字母之外的字符，并替换为空字符串
    cleaned_text = re.sub('[^A-Z0-9]', '', text)
    return cleaned_text

char_set={}
ascii=65
digits=10
for i in range(26):
    if str(digits)[-1]==str(digits)[-2]:
        digits+=1
    char_set.update({chr(ascii):digits})
    ascii+=1
    digits+=1

def image_to_text(text):       
    #將辨識出的字組合在一起
    predict=''
    for i in text:
        predict+=i
    predict  

    # 過濾字串
    text=str.upper(predict)
    cleaned_text = remove_special_characters(text) 
    cleaned_text=cleaned_text[:10]
    # print(cleaned_text)

    #將字串轉成對應的數字，來預測第11個字
    container_text=[]
    for i in cleaned_text:
        if(str.isupper(i)):
            container_text.append(char_set[i])
        else:
            container_text.append(int(i))     

    #透過公式轉成對應的第11個數字
    sum=0
    for i in range(len(container_text)):
        sum+=container_text[i]*(2**i)
    number11=sum%11    
    if sum%11==10:
        number11=0
    cleaned_text+=str(number11)
    return cleaned_text    

import cv2

# 打开视频文件
video = cv2.VideoCapture('video/video_0001.avi')

# 检查视频文件是否成功打开
if not video.isOpened():
    print('無法打開文件')
    exit()

#判斷正確的有幾張
correct=0
#總共偵測到的有幾張
counts=0
number_set=[]
cv2.namedWindow("video",cv2.WINDOW_NORMAL)
cv2.resizeWindow('video', 800, 600)

#累計
count=0
#計算fps
counter = 0
fps = video.get(cv2.CAP_PROP_FPS) #視頻平均幀率
print('影片禎數：',fps)
start_time = time.time()
fps=''
# 循环读取视频帧
while True:
    count+=1
    if cv2.waitKey(1) == ord('q'):      # 每一毫秒更新一次，直到按下 q 結束
        break
    # 读取一帧图像
    ret, frame = video.read()
    counter += 1  # 計算幀數
    # 检查是否成功读取到图像
    if not ret:
        break
    #偵測
    r_image ,crop,label   = yolo.detect_image(Image.fromarray(frame))
    # print(r_image.size)
    #沒有偵測到或是信心度不足就下一貞
    if(crop==None or label<=0.7):
        r_image = cv2.cvtColor(np.array(r_image), cv2.COLOR_RGB2BGR)
        if time.time() - start_time >= 1:
            fps="FPS:{0}".format(float('%.1f' % (counter / (time.time() - start_time))))
            cv2.putText(r_image,fps,(5,40), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2 , cv2.LINE_AA)
            start_time = time.time()
            counter=0
        cv2.putText(r_image,fps, (5,40), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2 , cv2.LINE_AA)    
        cv2.imshow('video', r_image)
        continue
    else:
        counts+=1
        # print(label)
        # plt.imshow(r_image)  
        # plt.show()
        crop_picture=r_image.crop(crop)
        crop_picture = crop_picture.convert('L')
        # 对图像进行对比度增强
        crop_picture = enhance_contrast(crop_picture, alpha, beta)
        text=reader.readtext(np.array(crop_picture), detail = 0)
        # crop_picture = adjust_brightness(crop_picture,1) 
        # plt.imshow(crop_picture)  
        # plt.show()
        
        #預測第11個字
        final_text=image_to_text(text)
        cv_image = cv2.cvtColor(np.array(r_image), cv2.COLOR_RGB2BGR)
        cv2.putText(cv_image, final_text, (crop[0],crop[3]+30), cv2.FONT_HERSHEY_SIMPLEX,1, (187,41,187), 2 , cv2.LINE_AA)
        if time.time() - start_time >= 1:
            fps="FPS:{0}".format(float('%.1f' % (counter / (time.time() - start_time))))
            cv2.putText(cv_image,fps, (5,40), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2 , cv2.LINE_AA)
            start_time = time.time()
            counter=0
        cv2.putText(cv_image, fps, (5,40), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2 , cv2.LINE_AA)    
        # plt.imshow(r_image)  
        # plt.text(crop[0],crop[3]+50, final_text, color='purple', fontsize=10)
        # plt.show() 
        
        # 顯示圖像
        cv2.imshow('video', cv_image)

        print('辨識的車牌：',final_text)
        if(final_text=='SEKU5875349'):
            correct+=1
        number_set.append(final_text)
    # start_time = time.time()qq

            
print('總共幀數：',counts)
print('偵測到車牌中辨識正確車牌的準確率：',correct/counts)
print('經過多數決後產生的最終辨識結果：',max(set(number_set),key=number_set.count))
# 释放视频对象和关闭窗口
cv2.destroyAllWindows()
video.release()