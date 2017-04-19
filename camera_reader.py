# -*- coding:utf-8 -*-
import cv2
import numpy as np
import time
import os
import math
import sys

from owner_train import Model

def square(x):
	return x * x

def distance_sqr(x1, y1, x2, y2):
	return square(x1-x2) + square(x2-y2)

def face_beautify(image,cascade, cascade2, processed_image):
	for (x, y, w, h) in processed_image:
		image1 = image[y:y+h, x:x+w]
		image_high = image1
		eyes = cascade2.detectMultiScale(image1)
		for (ex, ey, ew, eh) in eyes:
			center_x = ex + ew * 0.5
			center_y = ey + eh * 0.5
			eyes1 = image1[ey:ey+eh, ex:ex+ew]
			eyes2 = eyes1
			kernel_radius = min(ew, eh) * 0.4
			for r in range(eh):
				for c in range(ew):
					diff_x = c - ew*0.5
					diff_y = r - eh*0.5
					distance = math.sqrt(diff_x * diff_x + diff_y * diff_y)
					p_x = 0
					p_y = 0
					if distance <= kernel_radius:
						re = (1 - math.cos(distance / kernel_radius * 2 * math.pi)) * 2.5
						p_x = -diff_x * (re / kernel_radius)
						p_y = -diff_y * (re / kernel_radius)
					if p_x < 0 : 
						p_x  = 0
					if p_y < 0 : 
						p_y = 0
					eyes2[r,c] = eyes1[int(r + p_y),int(c + p_x)]
			image1[ey:ey+eh, ex:ex+ew] = eyes2	
		image_high1 = cv2.bilateralFilter(image_high, 15, 37, 37)
		#image_high2 = image_high1 - image1 + 128 
		image_high3 = cv2.GaussianBlur(image_high1,(1, 1),0)
		#image_high4 = image1 + 2 * image_high3 - 255
		#final = image1 * 0.45 + image_high4 * 0.55
		c_x = x + w * 0.5
		c_y = y + h * 0.5
		radius = min(w, h) * 2
		image_high4 = image_high3
		for row in range(h):
			for col in range(w):
				diff_x = col - w * 0.5
				diff_y = col - h * 0.5
				distance = math.sqrt(square(col - w*0.5) + square(row - h*0.5))
				m_x = 0
				m_y = 0
				if distance <= radius:
					re = (1 - math.cos(distance / radius * 2 * math.pi)) * 2
					m_x = -diff_x * (re / radius)
					m_y = -diff_y * (re / radius)
				if m_x < 0:
					m_x = 0
				if m_y < 0:
					m_y = 0
				image_high4[row,col] = image_high3[int(row + m_y), int(col + m_x)]
		image[y:y+h, x:x+w] = image_high4
	return image

if __name__ == '__main__':
	cap = cv2.VideoCapture(0)
	face_cascade = cv2.CascadeClassifier("/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")
	eye_cascade = cv2.CascadeClassifier('/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_eye.xml')
	model = Model()
	model.load()
	while(True):
		ret, frame = cap.read() # 读取每一帧，ret返回是否成功，frame返回图像本身

		frame = cv2.flip(frame, 1) # flip the image，让图像变成镜面
		# 灰度变换
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Haar级联检测 face-cascade
		cascade = face_cascade
		# 直方图均衡化
		equalImage = cv2.equalizeHist(frame_gray)
		# 人脸识别
		facerect = cascade.detectMultiScale(equalImage, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))
		#facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.01, minNeighbors=3, minSize=(3, 3))

		if len(facerect) > 0:
			print('face detected')
			color = (255, 255, 255)  # 白
			# 裁剪
			for rect in facerect:
				# 创建围绕检测的面部的矩形
				#cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), color, thickness=2)

				x, y = rect[0:2]
				width, height = rect[2:4]
				image = frame[y - 10: y + height, x: x + width]

				result = model.predict(image)
				if result == 0:  # owner
					print('Hi Shixin!')
					save_image = face_beautify(frame, face_cascade, eye_cascade, facerect)
					cv2.imshow('image_detect', frame)
					k = cv2.waitKey(10)

					if k == ord('s'):
						cv2.imwrite('/Users/gushixin/Desktop/new.jpg', save_image)
						break
				else:
					print('Hello Stranger!')
		
			# 10msec等待键入
			k = cv2.waitKey(10)
			#Esc 退出
			if k == 27:
			#sys.exit(0)
				break

	#退出拍摄
	cap.release()
	cv2.destroyAllWindows()
