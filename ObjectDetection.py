import torch
import cv2 as cv
import numpy as np
import pandas
import json
import sys

model=torch.hub.load('C:/Users/Reinaldo yang/Programming/research/yolov5-master/', 'custom',source='local', path='C:/Users/Reinaldo yang/Programming/research/new_best.pt')
video=cv.VideoCapture(0)

# Function to draw Centroids on the deteted objects and returns updated image
def draw_centroids_on_image(output_image, json_results):   
    data = json.loads(json_results) # Converting JSON array to Python List
    # Accessing each individual object and then getting its xmin, ymin, xmax and ymax to calculate its centroid
    for objects in data:
        xmin = objects["xmin"]
        ymin = objects["ymin"]
        xmax = objects["xmax"]
        ymax = objects["ymax"]
        
        #print("Object: ", data.index(objects))
        #print ("xmin", xmin)
        #print ("ymin", ymin)
        #print ("xmax", xmax)
        #print ("ymax", ymax)
        
        #Centroid Coordinates of detected object
        cx = int((xmin+xmax)/2.0)
        cy = int((ymin+ymax)/2.0)   
        #print(cx,cy)
    
        cv.circle(output_image, (cx,cy), 2, (0, 0, 255), 2, cv.FILLED) #draw center dot on detected object
        cv.putText(output_image, str(str(cx)+" , "+str(cy)), (int(cx)-40, int(cy)+30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv.LINE_AA)

    return (output_image)
        

#check whether cap is opened
while (video.isOpened()):
    ret, frame = video.read()

    k = cv.waitKey(5)
    if k == 27: #exit by pressing Esc key
        cv.destroyAllWindows()
        sys.exit()

    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #make detections    
    results=model(frame)
    
    # cv.imshow('Detection_Fixed', frame)

    # cv.imshow('YOLO', np.squeeze(results.render()))
    results.xyxy[0]
    print(results.pandas().xyxy[0])

    #Results in JSON
    json_results = results.pandas().xyxy[0].to_json(orient="records") # im predictions (JSON)
                #print(json_results)
                
    results.render()  # updates results.imgs with boxes and labels                    
    output_image = results.ims[0] #output image after rendering
    output_image = cv.cvtColor(output_image, cv.COLOR_RGB2BGR)
                
    output_image = draw_centroids_on_image(output_image, json_results) # Draw Centroids on the deteted objects and returns updated image
                
    cv.imshow("Output", output_image) #Show the output image after rendering
                #cv2.waitKey(1)
