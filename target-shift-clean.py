import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
import glob
import time

max_binary_value = 255

threshold = 'threshold'
angle = 'angle'
votes = 'votes'
minPix = 'min pix'
maxGap = 'max gap'

angle_eps_c = 'angle eps'
dist_eps_c = 'dist eps'
merge_eps_c = 'merge eps'

window_name = 'Target Shift'
kernel2 = np.ones((2,2), np.uint8)

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
for img in glob.glob("./new-train/*.jpg"):
##for img in glob.glob("./double-train/*.jpg"):
    count += 1
    n = cv2.imread(img)[305:605, 655: 955]
    allIm.append(n)
##    if count > 50: break
orig = allIm[0]
image_value = "image"
image_val_max = len(allIm) - 1

## [Threshold_Demo]
def Threshold_Demo(val):
    startTime = time.time()
    
    image_num = cv2.getTrackbarPos(image_value, window_name)

    orig = allIm[image_num]
    src_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    #add a mask on progress
    mask_c = np.ones((orig.shape[0], orig.shape[1]), np.uint8)
    cv2.circle(mask_c, (145, 145), 125, (0,0,0), thickness=12)  #mask on progress ring
    cv2.circle(mask_c, (145, 145), 50, (0,0,0), thickness=-1)   #mask on center reticle
    src_gray = cv2.bitwise_and(src_gray, src_gray, mask=mask_c)

    cv2.imshow("orig", orig)

    threshold_value = cv2.getTrackbarPos(threshold, window_name) + 230
    angle_value = cv2.getTrackbarPos(angle, window_name)
    votes_value = cv2.getTrackbarPos(votes, window_name)
    minPix_value = cv2.getTrackbarPos(minPix, window_name)
    maxGap_value = cv2.getTrackbarPos(maxGap, window_name)
    
    _, dst = cv2.threshold(src_gray, threshold_value, max_binary_value, 0 )
    cv2.imshow(window_name, dst)

    dil = cv2.dilate(dst, kernel2, iterations=3)
##    cv2.imshow("dil", dil)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / angle_value  # angular resolution in radians of the Hough grid
    threshold_votes = votes_value  # minimum number of votes (intersections in Hough grid cell)
##    min_line_length = minPix_value  # minimum number of pixels making up a line
##    max_line_gap = maxGap_value  # maximum gap in pixels between connectable line segments

    origin_x = 144
    origin_y = 143
    
    slope_eps = cv2.getTrackbarPos(angle_eps_c, window_name)
    dist_eps = cv2.getTrackbarPos(dist_eps_c, window_name)
    merge_eps = cv2.getTrackbarPos(merge_eps_c, window_name)

    houghStart = time.time()
    
    houghIm = cv2.cvtColor(dst.copy(), cv2.COLOR_GRAY2RGB)
    houghFiltered = houghIm.copy()
    houghRes1 = []
    lines2 = cv2.HoughLines(dil, rho, np.pi / 180, 25)   #90 for 2 deg
    if lines2 is not None:
##        print("found hough")
        for i in range(0, len(lines2)):
            rho = lines2[i][0][0]
            theta = lines2[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

            if theta != 0:
                slope = -1/math.tan(theta)#-a/b
                inter = y0 - slope * x0
                if abs((origin_x * slope + inter) - origin_y) < merge_eps:
##                    print("near origin", slope, rho, theta, math.degrees(theta))
                    cv2.line(houghIm, pt1, pt2, (0,255,0), 1, cv2.LINE_AA)
                    houghRes1.append([theta, rho])
            elif theta == 0 and abs(rho - origin_x) < merge_eps:
##                print("vertical near origin")
                cv2.line(houghIm, pt1, pt2, (255,0,0), 1, cv2.LINE_AA)
                houghRes1.append([math.pi, rho])
        cv2.circle(houghIm, (origin_x, origin_y), merge_eps, (0,0,255), 1)
    cv2.imshow("hough", houghIm)

    houghRes1.sort()

    print("Hough identification time:", time.time()-houghStart)
        
##    print(len(houghRes1))
##    print(houghRes1)

    #preprocess v2- search from end to find if segments exist within dist_eps; also serves as finding the smaller target
    numTargets = 1
    houghRes2 = []  #this stores the endpoints of targets
    stop = -1
    cur = math.degrees(houghRes1[-1][0])
    for x in range(len(houghRes1) - 2, 0, -1):
        if abs(math.degrees(houghRes1[x][0]) - cur) > dist_eps:
            numTargets += 1
            cur = math.degrees(houghRes1[x][0])
            if stop == -1:
                stop = x
##    print("preprocessing done: number of targets =", numTargets)
    innerStart = time.time()
    prevBin = 0
    if numTargets == 2:
        for x in range(1, len(houghRes1)):
            if abs(math.degrees(houghRes1[x][0]) - math.degrees(houghRes1[x-1][0])) < slope_eps and x != len(houghRes1)-1:
                continue
            else:
                avgT = 0
                avgR = 0
                count= 0
                z = x
                if x == len(houghRes1)-1:
                    if abs(math.degrees(houghRes1[x][0]) - math.degrees(houghRes1[x-1][0])) < slope_eps:
##                        print("grouping all")
                        z += 1
                    else:
##                        print("grouping prev, and binning last")
##                        print(houghRes1[x][0], houghRes1[x][1])
                        houghRes2.append([houghRes1[x][0], houghRes1[x][1]])
                for y in range(prevBin, z):
                    #average and append to houghRes2
                    avgT += houghRes1[y][0]
                    avgR += houghRes1[y][1]
                    count += 1
##                print(count, "in bin")
                houghRes2.append([avgT/count, avgR/count])
                prevBin = z
    else:
        #classify from stop to end the center
        maxDeg = houghRes1[-1][0]
        maxRad = houghRes1[-1][1]
        minDeg = houghRes1[stop + 1][0]
        minRad = houghRes1[stop + 1][1]
        houghRes2.append([(maxDeg+minDeg)/2, (maxRad+minRad)/2])
        
        for x in range(1, stop + 2):
    ##        print(math.degrees(houghRes1[x][0]), houghRes1[x][1], math.degrees(houghRes1[x-1][0]), houghRes1[x-1][1])
            if abs(math.degrees(houghRes1[x][0]) - math.degrees(houghRes1[x-1][0])) < slope_eps and x != stop + 1:
                continue
            else:
                avgT = 0
                avgR = 0
                count = 0
                z = x
                if x == stop + 1:
##                    print('last')
                    if abs(math.degrees(houghRes1[x][0]) - math.degrees(houghRes1[x-1][0])) < slope_eps:
##                        print("grouping all")
                        z += 1
                    else:
##                        print("grouping prev, and binning last")
                        houghRes2.append([houghRes1[x][0], houghRes1[x][1]])
                for y in range(prevBin, z):
                    #average and append to houghRes2
                    avgT += houghRes1[y][0]
                    avgR += houghRes1[y][1]
                    count += 1
##                print(count, "in bin")
                houghRes2.append([avgT/count, avgR/count])
                prevBin = z
        #bin last
##    print()
##    print(len(houghRes2))
    print("classification runtime:", time.time()-innerStart)

    for x in range(len(houghRes2)):
        if houghRes2[x][0] == math.pi:
            houghRes2[x][0] = 0
        a = math.cos(houghRes2[x][0])
        b = math.sin(houghRes2[x][0])
        x0 = a * houghRes2[x][1]
        y0 = b * houghRes2[x][1]
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(houghFiltered, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)
##    print(houghRes2)

    #now find center(s) of target(s)
    center = []
    if len(houghRes2) == 2:
        #cartesian conversion
        a = math.cos(houghRes2[0][0])
        b = math.sin(houghRes2[0][0])
        x0 = a * houghRes2[0][1]
        y0 = b * houghRes2[0][1]
        if x0 + 1000 * (-b) <= origin_x:
            x0 = x0 - 1000 * (-b)
            y0 = y0 - 1000 * a
        else:
            x0 = x0 + 1000 * (-b)
            y0 = y0 + 1000 * a
        a = math.cos(houghRes2[1][0])
        b = math.sin(houghRes2[1][0])
        x1 = a * houghRes2[1][1]
        y1 = b * houghRes2[1][1]
        if x1 + 1000 * (-b) <= origin_x:
            x1 = x1 - 1000 * (-b)
            y1 = y1 - 1000 * a
        else:
            x1 = x1 + 1000 * (-b)
            y1 = y1 + 1000 * a
        avgAngle = (math.atan2(y0 - origin_y, x0 - origin_x) + math.atan2(y1 - origin_y, x1 - origin_x)) / 2
        center = [avgAngle]
    elif len(houghRes2) == 3 or len(houghRes2) == 4:
        #use largest angle
        houghRes2.sort()
        a = math.cos(houghRes2[-1][0])
        b = math.sin(houghRes2[-1][0])
        x1 = a * houghRes2[-1][1]
        y1 = b * houghRes2[-1][1]
        if x1 + 1000 * (-b) <= origin_x:
            x1 = x1 - 1000 * (-b)
            y1 = y1 - 1000 * a
        else:
            x1 = x1 + 1000 * (-b)
            y1 = y1 + 1000 * a
        center = [math.atan2(y1 - origin_y, x1 - origin_x)]
        
    else:
        print("bad classification")

    if len(center) != 0:
        radius = 100
        x = round(origin_x + math.degrees(math.cos(center[0])) * radius)
        y = round(origin_y + math.degrees(math.sin(center[0])) * radius)
        cv2.line(houghFiltered, (origin_x, origin_y), (x,y), (255,0,0), 1, cv2.LINE_AA)
    
    cv2.imshow("hough filtered", houghFiltered)
    print("done with hough; time to run:", time.time() - startTime)
    print()

## [window]
# Create a window to display results
cv2.namedWindow(window_name)
## [window]

## [trackbar]
cv2.createTrackbar(image_value, window_name, 16, image_val_max, Threshold_Demo)

cv2.createTrackbar(threshold, window_name, 7, 25, Threshold_Demo)
cv2.createTrackbar(angle, window_name, 180, 360, Threshold_Demo)
cv2.createTrackbar(votes, window_name, 8, 25, Threshold_Demo)
cv2.createTrackbar(minPix, window_name, 5, 30, Threshold_Demo)
cv2.createTrackbar(maxGap, window_name, 5, 25, Threshold_Demo)

cv2.createTrackbar(angle_eps_c, window_name, 2, 45, Threshold_Demo)
cv2.createTrackbar(dist_eps_c, window_name, 11, 45, Threshold_Demo)
cv2.createTrackbar(merge_eps_c, window_name, 8, 20, Threshold_Demo)


# Call the function to initialize
Threshold_Demo(0)
# Wait until user finishes program
cv2.waitKey()
