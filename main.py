import cv2 as cv
import numpy as np
import glob

#define dimension of chess board
chessboardSize=(5,7)

#Array to store actual 3D points (real world space) for each checkerboard images
objPoints=[]
#Array to store actual 2D points (image points) for each checkerboard images
imgPoints=[]

#termination criteria (default from openCV)
criteria=(cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER,30,0.001)

#defining world coordinates for 3D points like (0,0,0),(1,0,0)...
objp = np.zeros((1, chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)*3

#extract path of individual images
images=glob.glob('./images/*jpg')

for image in images:
    # print(image)
    img=cv.imread(image)
    img = cv.resize(img, (0,0), fx=0.6, fy=0.6) 
    #convert to gray scale image
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #find the chess board corner
    ret,corners=cv.findChessboardCorners(gray,chessboardSize,cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
    
    
    if ret == True:
        objPoints.append(objp)
        #refining  2d subpixels
        corners2=cv.cornerSubPix(gray,corners,(11,11),(-1,-1), criteria)
        imgPoints.append(corners2)

        #draw and display the corners
        cv.drawChessboardCorners(img,chessboardSize,corners2,ret)
        cv.imshow('img',img)
        cv.waitKey(1000)

cv.destroyAllWindows

#CALIBRATION
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)
print("Camera Calibrated",ret)
print("\nCamera Matrix:\n", mtx)
print("\nDistortion Parameters\n", dist)
print("\nRotation Vectors\n", rvecs)
print("\nTranslation Vector\n", tvecs)

np.savez("CameraParameters",cameraMatrix=mtx,dist=dist,rvecs=rvecs,tvecs=tvecs)

mean_error = 0
for i in range(len(objPoints)):
    imgPoints2, _ = cv.projectPoints(objPoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgPoints[i], imgPoints2, cv.NORM_L2)/len(imgPoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objPoints)) )