import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
import glob

max_value = 255
max_type = 4
max_binary_value = 255
min_binary_value = 0
trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_type2 = 'Type2: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
trackbar_value2 = 'Value2'
window_name = 'Threshold Demo'
kernel = np.ones((2,2), np.uint8)
kernel3 = np.ones((3,3), np.uint8)
rot_mask = np.zeros((1,1), np.uint8)

##orig = cv2.imread("./train/20181121013009_1.jpg")[305:605, 655:955] #in prog early
##orig = cv2.imread("./train/20181121013019_1.jpg")[305:605, 655:955] #in prog after
##orig = cv2.imread("./train/20181121013018_1.jpg")[305:605, 655:955] #double
##orig = cv2.imread("./train/20181121015035_1.jpg")[305:605, 655:955] #overlap targets
##orig = cv2.imread("./train/20181121015221_1.jpg")[305:605, 655:955] #mixed back
##orig = cv2.imread("./train/20181121015223_1.jpg")[305:605, 655:955] #at end
##orig = cv2.imread("./train/20181121015422_1.jpg")[305:605, 655:955] #overlap with hit
##orig = cv2.imread("./train/20181121015443_1.jpg")[305:605, 655:955] #overlap with done (top is vis)
##orig = cv2.imread("./train/20181121015447_1.jpg")[305:605, 655:955] #overlap with done
##orig = cv2.imread("./train/20181121122156_1.jpg")[305:605, 655:955] #stalk overlap
##orig = cv2.imread("./train/20181121012950_1.jpg")[305:605, 655:955] #perfect hit

allIm = []
count = 0
for img in glob.glob("./train/*.jpg"):
    count += 1
    n = cv2.imread(img)[305:605, 655: 955]
    allIm.append(n)
    if count > 16: break
orig = allIm[0]
image_value = "image"
image_val_max = len(allIm) - 1

## [Threshold_Demo]
def Threshold_Demo(val):
    #0: Binary
    #1: Binary Inverted
    #2: Threshold Truncated
    #3: Threshold to Zero
    #4: Threshold to Zero Inverted
    image_num = cv2.getTrackbarPos(image_value, window_name)

    orig = allIm[image_num]
    ####rotational mask
    ##w = math.sqrt((19 * 19) + (2 * 2))
    w = 15
    ##h = math.sqrt((4 * 4) + (1 * 1))
    h = 6
    r = 118
    cx = 145
    cy = 145
    y = math.degrees(2 * math.asin((h/2)/r))
    x = (h - y) / 2
    z = 180 - ((180 - y) / 2) - 90
    rot_mask = np.zeros((orig.shape[0], orig.shape[1]), np.uint8)

    for deg in range(0, 360, 5):
        Ax = cx + r * math.cos(math.radians(x + deg))
        Ay = cy + r * math.sin(math.radians(y + deg))
        Bx = cx + r * math.cos(math.radians(x + y + deg))
        By = cy + r * math.sin(math.radians(x + y + deg))

        Cx = Ax + w * math.cos(math.radians(z + deg))
        Cy = Ay + w * math.sin(math.radians(z + deg))
        Dx = Bx + w * math.cos(math.radians(z + deg))
        Dy = Ay + w * math.sin(math.radians(z + deg))

        pts = np.array([[Ax,Ay], [Bx,By], [Cx, Cy], [Dx, Dy]], np.int32)
        rect= cv2.minAreaRect(pts)
        rot = cv2.boxPoints(rect)
        rot = np.int0(rot)
        cv2.drawContours(rot_mask, [rot], 0, (255,255,255), -1)
    mask_rot_img = cv2.bitwise_and(orig, orig, mask=rot_mask)
    img = cv2.cvtColor(mask_rot_img, cv2.COLOR_BGR2RGB)
    src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow("orig", orig)
    cv2.imshow("color", img)
    cv2.imshow("gray", src_gray)

    
    threshold_type = cv2.getTrackbarPos(trackbar_type, window_name)
    threshold_type2 = cv2.getTrackbarPos(trackbar_type2, window_name)
    threshold_value = cv2.getTrackbarPos(trackbar_value, window_name)
    threshold2 = cv2.getTrackbarPos(trackbar_value2, window_name)
    _, dst = cv2.threshold(src_gray, threshold_value, max_binary_value, threshold_type )
    _, dst = cv2.threshold(dst, threshold2, min_binary_value, threshold_type2)
    cv2.imshow(window_name, dst)
    blur = cv2.GaussianBlur(dst, (19,19), 0)
    cv2.imshow("blur", blur)

##    morph = cv2.dilate(dst, rot_mask, iterations=1)
    morph = cv2.erode(dst, kernel, iterations=1)
##    morph = cv2.dilate(morph, kernel, iterations=2)
##    morph = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)
    cv2.imshow("morph1", morph)

    flood = morph.copy()
    floodmask = np.zeros((dst.shape[0]+2, dst.shape[1]+2), np.uint8)
    cv2.floodFill(flood, floodmask, (0,0), 255)
    flood = cv2.bitwise_not(flood)
    floodout = morph | flood

    floodout = cv2.erode(floodout, kernel, iterations=1)
    floodout = cv2.dilate(floodout, kernel, iterations=2)
    cv2.imshow("flood", floodout)

    floodcont = cv2.cvtColor(floodout.copy(), cv2.COLOR_GRAY2RGB)
    im2, contours, hiearchy = cv2.findContours(floodout, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("image", image_num)
    print(contours)
    cv2.drawContours(floodcont, contours, -1, (0,0,255), 2)
    cv2.imshow("cont", floodcont)

    floodctr = floodcont.copy()
    avgCtr = np.zeros((len(contours), 2), np.float64)
    index = 0
    print()
    for contour in contours:
##        print(contour)
##        print(contour[:,0,0], np.mean(contour[:,0,0]))
##        print(contour[:,0,1], np.mean(contour[:,0,1]))
        print("avg")
        avgX = np.mean(contour[:,0,0])
        avgY = np.mean(contour[:,0,1])
        print(avgX, avgY)
        avgCtr[index] = np.array([avgX, avgY])
        floodctr[int(round(avgY)), int(round(avgX))] = (0,255,0)
        print(avgCtr[index])
        index += 1
    print("all")
    print(avgCtr)
##    cv2.drawContours(floodctr, avgCtr, -1, (0, 255, 0), 2)
    
    
    cv2.imshow("ctr", floodctr)
## [Threshold_Demo]

'''
####rotational mask
##w = math.sqrt((19 * 19) + (2 * 2))
w = 20
##h = math.sqrt((4 * 4) + (1 * 1))
h = 6
r = 116
cx = 145
cy = 145
y = math.degrees(2 * math.asin((h/2)/r))
x = (h - y) / 2
z = 180 - ((180 - y) / 2) - 90
rot_mask = np.zeros((orig.shape[0], orig.shape[1]), np.uint8)

for deg in range(0, 360, 5):
    Ax = cx + r * math.cos(math.radians(x + deg))
    Ay = cy + r * math.sin(math.radians(y + deg))
    Bx = cx + r * math.cos(math.radians(x + y + deg))
    By = cy + r * math.sin(math.radians(x + y + deg))

    Cx = Ax + w * math.cos(math.radians(z + deg))
    Cy = Ay + w * math.sin(math.radians(z + deg))
    Dx = Bx + w * math.cos(math.radians(z + deg))
    Dy = Ay + w * math.sin(math.radians(z + deg))

    pts = np.array([[Ax,Ay], [Bx,By], [Cx, Cy], [Dx, Dy]], np.int32)
    rect= cv2.minAreaRect(pts)
    rot = cv2.boxPoints(rect)
    rot = np.int0(rot)
    cv2.drawContours(rot_mask, [rot], 0, (255,255,255), -1)
mask_rot_img = cv2.bitwise_and(orig, orig, mask=rot_mask)
img = cv2.cvtColor(mask_rot_img, cv2.COLOR_BGR2RGB)
src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

####circular mask
##mask_c = np.zeros((orig.shape[0], orig.shape[1]), np.uint8)
##cv2.circle(mask_c, (145, 145), 125, (255,255,255), thickness=21)
##mask_img = cv2.bitwise_and(orig, orig, mask=mask_c)
##src_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

####color mask
##img = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
##src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("orig", orig)
cv2.imshow("color", img)
cv2.imshow("gray", src_gray)
'''

## [window]
# Create a window to display results
cv2.namedWindow(window_name)
## [window]

## [trackbar]
cv2.createTrackbar(image_value, window_name, 0, image_val_max, Threshold_Demo)
# Create Trackbar to choose type of Threshold
cv2.createTrackbar(trackbar_type, window_name , 3, max_type, Threshold_Demo)
cv2.createTrackbar(trackbar_type2, window_name ,4, max_type, Threshold_Demo)
# Create Trackbar to choose 1st Threshold value
cv2.createTrackbar(trackbar_value, window_name , 130, max_value, Threshold_Demo)    #145
cv2.createTrackbar(trackbar_value2, window_name , max_value, max_value, Threshold_Demo)   #190
## [trackbar]

# Call the function to initialize
Threshold_Demo(0)
# Wait until user finishes program
cv2.waitKey()
