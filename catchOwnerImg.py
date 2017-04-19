import numpy as np
import cv2
import time
import os

face_cascade = cv2.CascadeClassifier('/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
frame_num = 506
face_num = 1
# 检测出图片中的人脸，并用方框标记出来
def face_detector(image, cascade):
    global face_num #引用全局变量
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #灰度化图片 
    equalImage = cv2.equalizeHist(grayImage) #直方图均衡化
    faces = cascade.detectMultiScale(equalImage, scaleFactor=1.3, minNeighbors=3)
        
    for (x,y,w,h) in faces:
        #裁剪出人脸，单独保存成图片，注意这里的横坐标与纵坐标不知为啥颠倒了
        cv2.imwrite("/Users/gushixin/Desktop/OwnerSensor/faceOnly/owner/self_%s.png" %(face_num), image[y:y+h, x:x+w])
        #cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        face_num = face_num + 1
    return image

def catchOwner():
	global frame_num
	cap = cv2.VideoCapture(0)
	while(True):
		ret, img = cap.read()
		img = cv2.flip(img, 1) # flip the image，让图像变成镜面
		show_image = face_detector(img, face_cascade)
		cv2.imshow('image',show_image)
		
		k = cv2.waitKey(2)
		if k == ord('s'):
			cv2.imwrite('/Users/gushixin/Desktop/OwnerSensor/data/owner/catch%s.jpg' % frame_num,img)
			frame_num += 1
		elif k == 27:
			break 

	cap.release()
	cv2.destroyAllWindows()

#当该py文件被直接运行时，代码将被运行，当该py文件是被导入时，代码不被运行   
if __name__ == '__main__':   
    catchOwner()