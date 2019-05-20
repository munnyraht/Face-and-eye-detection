#!/usr/bin/env python
# coding: utf-8

# 
# # Live Face and Eye detection using open cv and haarcascades classifiers
# 
# 
# 

# In[11]:


import numpy as np
import cv2
 
face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')
    


def face_eye_detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    # When no faces detected, face_classifier returns and empty tuple
    if faces is ():
        pass
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(127,0,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_classifier.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)
    return image



cap= cv2.VideoCapture(0)
while True:
    ret,frame= cap.read()
    frame=cv2.flip(frame,1)
    cv2.imshow('face and eye detection',face_eye_detect(frame))
    if cv2.waitKey(1)==13: #13 is the enter key
        break
cap.release()
cv2.destroyAllWindows()
        

        
        


# In[ ]:




