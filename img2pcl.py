import cv2
import itertools
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pykitti
import utm

basedir = 'C:\\KITTI'
date = '2011_09_26'
drive = '0005'

# projection matrix after rectification
P_00 = np.asarray([[7.070912e+02, 0.000000e+00, 6.018873e+02, 0.000000e+00],
                   [0.000000e+00, 7.070912e+02, 1.831104e+02, 0.000000e+00],
                   [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]])
        
P_01 = np.asarray([[7.070912e+02, 0.000000e+00, 6.018873e+02, -3.798145e+02],
                  [0.000000e+00, 7.070912e+02, 1.831104e+02, 0.000000e+00],
                  [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]])

f =   P_01[0,0]
bx =  P_01[0,3]/f
cu =  P_01[0,2]
cv =  P_01[1,2] 
cus = P_00[0,2]

Q = np.asarray([[1, 0, 0,     -cu        ],
                [0, 1, 0,     -cv        ],
                [0, 0, 0,     f          ],
                [0, 0, -1/bx, (cu-cus)/bx]])
                
print(Q)
print('Length of baseline: '+str(bx))

for nr in range(0,9):
    # Load Images
    imgL = cv2.imread('C:/KITTI/2011_09_26/2011_09_26_drive_0013_sync/image_00/data/000000000'+str(nr)+'.png',cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread('C:/KITTI/2011_09_26/2011_09_26_drive_0013_sync/image_01/data/000000000'+str(nr)+'.png',cv2.IMREAD_GRAYSCALE)

    # calculate disparity
    stereo = cv2.StereoBM_create(numDisparities=128, blockSize=25)
    '''    
    # https://github.com/opencv/opencv/blob/master/samples/python/stereo_match.py
    # https://stackoverflow.com/questions/31308641/c-opencv-depth-map-issue-items-in-the-point-cloud-have-some-distortions
    window_size = 3
    min_disp = 16
    num_disp = 128
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 16,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )        
    '''    
    
    disparity = stereo.compute(imgL,imgR)
    fig = plt.imshow(disparity, cmap='hot')

    # calculate pointcloud
    pcl = cv2.reprojectImageTo3D(disparity,Q,handleMissingValues=True)
    pointlist = []
    for i in range(0,pcl.shape[0]):
        for j in range(0,pcl.shape[1]):
            pointlist.append(pcl[i,j])        
    pcl = np.vstack(pointlist)

    # filter values which are to big
    binary = pcl[:,2]<1000 
    binary3 = np.column_stack((binary,binary,binary)) 
    pclFiltered = pcl[binary3]
    pclFiltered = np.reshape(pclFiltered,(-1,3))

    # write pointcloud
    np.savetxt('pcl_'+str(nr)+'.txt',pclFiltered,delimiter=',',fmt='%1.7f')