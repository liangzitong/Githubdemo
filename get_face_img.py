import numpy as np
import cv2
import time
import os

face_cascade = cv2.CascadeClassifier('/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
face_num0 = 1000
face_num1 = 0
images = []
labels = []
def traverse_dir(path):
    for file_or_dir in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))
        print(abs_path)
        if os.path.isdir(abs_path):  # dir
            traverse_dir(abs_path)
        else:                        # file
            if file_or_dir.endswith('.jpg') or file_or_dir.endswith('.jpeg') or file_or_dir.endswith('.png'):
                image = read_image(abs_path)
                images.append(image)
                labels.append(path)

    return images, labels
def read_image(file_path):
    image = cv2.imread(file_path)

    return image
def extract_data(path):
    images, labels = traverse_dir(path)
    images = np.array(images)
    labels = np.array([0 if label.endswith('owner') else 1 for label in labels])

    return images, labels

def face_detector1(image, cascade):
    global face_num1 #引用全局变量
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #灰度化图片 
    equalImage = cv2.equalizeHist(grayImage) #直方图均衡化
    faces = cascade.detectMultiScale(equalImage, scaleFactor=1.3, minNeighbors=3)
        
    for (x,y,w,h) in faces:
        #裁剪出人脸，单独保存成图片，注意这里的横坐标与纵坐标不知为啥颠倒了
        cv2.imwrite("/Users/gushixin/Desktop/OwnerSensor/faceOnly/other/other_%s.png" %(face_num1), image[y:y+h, x:x+w])
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        face_num1 = face_num1 + 1
    return image

def face_detector0(image, cascade):
    global face_num0 #引用全局变量
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #灰度化图片 
    equalImage = cv2.equalizeHist(grayImage) #直方图均衡化
    faces = cascade.detectMultiScale(equalImage, scaleFactor=1.3, minNeighbors=3)
        
    for (x,y,w,h) in faces:
        #裁剪出人脸，单独保存成图片，注意这里的横坐标与纵坐标不知为啥颠倒了
        cv2.imwrite("/Users/gushixin/Desktop/OwnerSensor/faceOnly/owner/owner_%s.png" %(face_num0), image[y:y+h, x:x+w])
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        face_num0 = face_num0 + 1
    return image

images, labels = traverse_dir('/Users/gushixin/Desktop/OwnerSensor/data/owner/')
for image in images:
    face_detector0(image,face_cascade)
images = []
images, labels = traverse_dir('/Users/gushixin/Desktop/OwnerSensor/data/other/')
for image in images:
    face_detector1(image, face_cascade)
