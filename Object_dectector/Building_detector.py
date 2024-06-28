# Import packages
import os
import cv2
import numpy as np
import sys
import glob
import random
import importlib.util
from tensorflow.lite.python.interpreter import Interpreter

import tensorflow as tf
import collections
import matplotlib.pyplot as plt


#%matplotlib inline

# This function choose image's central box 
def area_central_box(boxes, imH, imW, idx):
    ymin_c = int(max(1,(boxes[idx][0] * imH)))
    xmin_c = int(max(1,(boxes[idx][1] * imW)))
    ymax_c = int(min(imH,(boxes[idx][2] * imH)))
    xmax_c = int(min(imW,(boxes[idx][3] * imW)))
    yc = ymax_c - ymin_c
    xc = xmax_c - xmin_c
    area_c = yc*xc 
    return area_c

def building_detector(modelpath, imgpath, lblpath, min_conf=0.5, savepath='C:/', txt_only=False):

    # Load the label map into memory
    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the Tensorflow Lite model into memory
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    float_input = (input_details[0]['dtype'] == np.float32)


    input_mean = 127.5
    input_std = 127.5

    image_path = imgpath    
    # Load image and resize to expected shape [1xHxWx3]
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape 
    
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects
           
    all_boxes = boxes
    all_classes = classes
    all_scores = scores 
    
    # Calculate the center point of each bounding box
    centers_b = (boxes[:, :2] + boxes[:, 2:]) / 2
    centers = centers_b[:,1]
    
    # Calculate the distance between each center point and the center of the image
    #image_center = np.array([image.width/ 2, image.height/ 2])
    image_center = 0.5
    distances = np.absolute(centers - image_center)

    detections = []
    
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    dist_ref = 1
    n_boxes = collections.defaultdict(list)
    pond = collections.defaultdict(list)
    
    try:
        # Find central box
        for i in range(len(scores)):
            if classes[i] == 0 and scores[i]> 0.6:
                n_boxes["ID"].append(i)

        dist_ref = 1
        cent_d = collections.defaultdict(list)
        for i in n_boxes["ID"]:
            dist_cent = distances[i]
            if dist_cent < dist_ref:
                dist_ref = distances[i]
                cent_d["central"].append(i)
                idx = i
        
        # Central box's area filter 
        area_c = area_central_box(boxes, imH, imW, idx)
        if area_c < 0.25*(imH*imW) and len(cent_d["central"]) > 1:
            idx = int(cent_d["central"][-2])
            area_c = area_central_box(boxes, imH, imW, idx)
            if area_c < 0.25*(imH*imW):
                idx = int(cent_d["central"][-1])
                area_c = area_central_box(boxes, imH, imW, idx)
                if area_c < 0.075*(imH*imW):
                    idx = int(cent_d["central"][-2])

        # Note: Several variables are needed, so, do not the function
        ymin_c = int(max(1,(boxes[idx][0] * imH)))
        xmin_c = int(max(1,(boxes[idx][1] * imW)))
        ymax_c = int(min(imH,(boxes[idx][2] * imH)))
        xmax_c = int(min(imW,(boxes[idx][3] * imW)))
        yc = ymax_c - ymin_c
        xc = xmax_c - xmin_c
        area_c = yc*xc  
        
        # Problem of pictures with buildings away from the place where are taken
        if xc >= 0.95*imW and yc <= 0.6*imH and ymax_c >= 0.95*imH:
            try:
                if idx == 0:
                    idx = 1
                else:
                    idx = int(cent_d["central"][-2])
            except:
                pass

        # Calculate if there are one or more box inside the central box
        for i in n_boxes["ID"]:           
            box_y = centers_b[i][0]*imH
            box_x = centers_b[i][1]*imW

            c1 = xmin_c < box_x and xmax_c > box_x
            c2 = ymin_c < box_y and ymax_c > box_y
            if c1 and c2:
                if i != idx:
                    pond["id"].append(i)

        dist_ref = 1
        
        # Choose a smaller box within the central box
        if len(pond["id"]) > 1:
            for j in range(len(pond["id"])):
                aux = pond["id"][j]
                y_b = int(min(imH,(boxes[aux][2] * imH))) - int(max(1,(boxes[aux][0] * imH)))
                x_b = int(min(imW,(boxes[aux][3] * imW))) - int(max(1,(boxes[aux][1] * imW)))
                area_b = y_b*x_b

                if area_b < area_c and area_b > 0.4*area_c and area_b > 0.2*(imH*imW):
                    dist_cent = distances[aux]
                    if dist_cent < dist_ref:
                        dist_ref = distances[aux]
                        idx = aux
        else:
            if len(pond["id"]) > 0:
                one_b = pond["id"][0]
                y_b = int(min(imH,(boxes[one_b][2] * imH))) - int(max(1,(boxes[one_b][0] * imH)))
                x_b = int(min(imW,(boxes[one_b][3] * imW))) - int(max(1,(boxes[one_b][1] * imW)))
                area_b = y_b*x_b
                if area_b < area_c and area_b > 0.6*area_c:
                    idx = one_b
        try:          
            # Check score and class
            if scores[idx] < 0.6 or classes[idx] == 1:
                idx = None
            # Check mimimun area to aims of avoid a wrong box 
            area_c = area_central_box(boxes, imH, imW, idx)
            if area_c < 0.075*(imH*imW):
                por_area = area_c/(imH*imW)*100
                print("area problem: %.2f" %por_area)
                idx = None
            
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[idx][0] * imH)))
            xmin = int(max(1,(boxes[idx][1] * imW)))
            ymax = int(min(imH,(boxes[idx][2] * imH)))
            xmax = int(min(imW,(boxes[idx][3] * imW)))


            ###################
            ###################
            sw = True
            cont = 1
            while sw == True:
                if image_path[-cont] == str("/") or image_path[-cont] == str("\\"):
                    image_id = image_path[-cont+1:-4]
                    sw = False
                cont+=1

            # Crop the image using NumPy slicing.
            cropped_image = image[ymin:ymax, xmin:xmax]

            # Preprocess the cropped image using the same preprocessing steps you used when training the TFLite model.
            input_shape = ((xmax-xmin),(ymax-ymin))
            image_resized = cv2.resize(cropped_image, input_shape)
            image_normalized = (image_resized.astype(np.float32) / 255)
            image_crop = cv2.cvtColor(image_normalized,cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(6,8))
            plt.imshow(image_crop)
            plt.show()  

            folder = 'Cropped_Images'
            if not os.path.exists(folder):
                os.makedirs(folder)
            path_dw = folder
            #print("Donwload: "+savepath+"/"+image_id+"_cropped.jpg")
            cv2.imwrite(savepath+"/"+image_id+".jpg", cropped_image)
          
            #cv2.imwrite(image_id+".jpg", cropped_image)

            ###################
            ###################

            thickness_box = 5      
            color_list = [
                (0, 0, 255),   # blue
                (0, 255, 0),   # Green
                (255, 0, 0),   # red
                (0, 255, 255), # Yellow
                (255, 0, 255), # Magenta
                (255, 255, 0), # Cyan
                (0, 0, 128),   # Dark Red
                (0, 128, 0),   # Dark Green
                (128, 0, 0),   # Dark Blue
                (0, 128, 128), # Dark Yellow
                (128, 0, 128), # Dark Magenta
                (128, 128, 0), # Dark Cyan
                (0, 0, 0),     # Black
                (255, 255, 255) # White
                ]
               
            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), color_list[0], thickness_box)

                       
            # Draw label
            object_name = labels[int(classes[idx])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[idx]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            detections.append([object_name, scores[idx], xmin, ymin, xmax, ymax])


            # All the results have been drawn on the image, now display the image
            if txt_only == False: # "text_only" controls whether we want to display the image results or just save them in .txt files
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(6,8))
                plt.imshow(image)
                plt.show()

        except:
            print("Foto", image_path)
            
        
        
        
        detections = []
        for i in range(len(all_scores)):
            if (all_scores[i] > min_conf):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(all_boxes[i][0] * imH)))
                xmin = int(max(1,(all_boxes[i][1] * imW)))
                ymax = int(min(imH,(all_boxes[i][2] * imH)))
                xmax = int(min(imW,(all_boxes[i][3] * imW)))

                cv2.rectangle(image, (xmin,ymin), (xmax,ymax), color_list[2], thickness_box)

                # Draw label
                object_name = labels[int(all_classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(all_scores[i]*100)) # Example: 'person: 72%'
                fontScale_label, thickness_label =  0.7, 2
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontScale_label, thickness_label) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                detections.append([object_name, all_scores[i], xmin, ymin, xmax, ymax])
        
        # All the results have been drawn on the image, now display the image
        if txt_only == False: # "text_only" controls whether we want to display the image results or just save them in .txt files
            plt.figure(figsize=(6,8))
            plt.imshow(image)
            plt.show()
            
            
    except:
        print("Foto", image_path)
        
    return