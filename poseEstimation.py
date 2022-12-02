import cv2 as cv
import numpy as np
import glob

#Load saved camera calibration data
with np.load('CameraParameters.npz') as file:
    mtx,dist,rvecs,tvecs=[file[i] for i in ('cameraMatrix','dist','rvecs','tvecs')]

#create a draw function that takes corneres in the chessboard and asxis points to draw a 3d axis
def draw(img,corners,imgpts):
    corner=tuple(corners[0].ravel())
    corner=[int(i) for i in corner]
    imgpts = np.int32(imgpts).reshape(-1,2)
    img=cv.line(img,corner,tuple(imgpts[0].ravel()),(255,0,0),10)
    img=cv.line(img,corner,tuple(imgpts[1].ravel()),(0,255,0),10)
    img=cv.line(img,corner,tuple(imgpts[2].ravel()),(0,0,255),10)
    return img

def drawBoxes(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

criteria=(cv.TERM_CRITERIA_EPS+cv.TermCriteria_MAX_ITER,30,0.001)

objp=np.zeros((9*7,3),np.float32)
objp[:,:2]=np.mgrid[0:7,0:9].T.reshape(-1,2)*2
#draw axis length of 3
axis=np.float32([[3,0,0],[0,3,0],[0,0,-3]]).reshape(-1,3)
axisBoxes = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

for image in glob.glob('*jpg'):
    img=cv.imread(image)
    img = cv.resize(img, (0,0), fx=0.6, fy=0.6) 

    #convert to grayscale
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners=cv.findChessboardCorners(gray,(7,9),None)

    if ret==True:

        corners2=cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        
        #find the rotation and translation vectors
        ret,rvecs,tvecs=cv.solvePnP(objp,corners2,mtx,dist)
        print("\ntransitional vector\n",tvecs)
        print("rotational vectors\n",rvecs)
        
        #project 3D points to image plane
        imgpts,jac=cv.projectPoints(axisBoxes,rvecs,tvecs,mtx,dist)
        
        img=drawBoxes(img,corners2,imgpts)

        rotM = cv.Rodrigues(rvecs)[0]
        cameraPosition = -np.matrix(rotM).T * np.matrix(tvecs)
        print("Camera Position\n",cameraPosition)

        cv.imshow('img',img)
        k = cv.waitKey(1000) & 0xFF
        if k==ord('s'):
            cv.imwrite('pose'+image,img)
cv.destroyAllWindows()