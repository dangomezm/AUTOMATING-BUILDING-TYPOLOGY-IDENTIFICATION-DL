from ultralytics import YOLO
import cv2
import os
from Perspective_Correction import *

# Keypoints model data
keypoint_model_path = 'D:/last_vf.pt'
model = YOLO(keypoint_model_path)

# Data
path = "D:/Image_Data"
data_path = path+"/OD_Story_Images_Masonry"
story_image = os.listdir(data_path)

#%%
for j in story_image:
    image_path = data_path+"/"+j
    img = cv2.imread(image_path)
    
    try:
        results = model(image_path)[0]
        
        keypoints = results[1]
        corners = keypoints.keypoints.data[0]
        corners = corners.numpy()
        saved_path = "D:/Image_Data/Keypoints_Images/"+j
        img_transf = corner_detection(corners, image_path, saved_path)
    except:
        try:
            results = model(image_path)[0]
            
            keypoints = results[0]
            corners = keypoints.keypoints.data[0]
            corners = corners.numpy()
            saved_path = "D:/Image_Data/Keypoints_Images/"+j
            img_transf = corner_detection(corners, image_path, saved_path)
        except: 
            pass
