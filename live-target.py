import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from PIL import ImageGrab
import time

import win32api
from ctypes import windll, Structure, c_long, byref

class POINT(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]

def getMousePos():
    pt = POINT()
    windll.user32.GetCursorPos(byref(pt))
    return pt.x, pt.y

leftState = win32api.GetKeyState(0x01)


kernel2 = np.ones((2,2), np.uint8)
origin_x = 144
origin_y = 143
slope_eps = 2
dist_eps = 11
merge_eps = 8

def analyze(position):
    img = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(655, 305, 955, 605))), cv2.COLOR_BGR2GRAY)
    mask_c = np.ones((img.shape[0], img.shape[1]), np.uint8)
    cv2.circle(mask_c, (145, 145), 125, (0,0,0), thickness=12)  #mask on progress ring
    cv2.circle(mask_c, (145, 145), 50, (0,0,0), thickness=-1)   #mask on center reticle
    src_gray = cv2.bitwise_and(img, img, mask=mask_c)
##    cv2.imshow("gray", src_gray)    #grayed image with mask
    _, dst = cv2.threshold(src_gray, 237, 255, cv2.THRESH_BINARY)    #230-245
##    cv2.imshow("prog", dst)     #thresholded image

    dil = cv2.dilate(dst, kernel2, iterations=3)    #dilate image

    houghIm = cv2.cvtColor(dst.copy(), cv2.COLOR_GRAY2RGB)
    houghFiltered = houghIm.copy()
    houghRes1 = []
    lines2 = cv2.HoughLines(dil, 1, np.pi / 180, 25)   #90 for 2 deg
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
    else:
        print("no mining detected")
##                time.sleep(1)
        return
    cv2.imshow("hough", houghIm)
    if len(houghRes1) == 0:
        return
    houghRes1.sort()

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

    #draw current position
    x = round(origin_x + math.degrees(math.cos(math.radians(position + 90))) * 100)
    y = round(origin_x + math.degrees(math.sin(math.radians(position + 90))) * 100)
    cv2.line(houghFiltered, (origin_x, origin_y), (x,y), (0,255,0), 1, cv2.LINE_AA)

    if len(center) != 0:
        radius = 100
        x = round(origin_x + math.degrees(math.cos(center[0])) * radius)
        y = round(origin_y + math.degrees(math.sin(center[0])) * radius)
        cv2.line(houghFiltered, (origin_x, origin_y), (x,y), (255,0,0), 1, cv2.LINE_AA)
    
    cv2.imshow("hough filtered", houghFiltered)

    return center

pos = 0
while (True):   #perhaps add a sleep?
    #if key pressed start loooking, otherwise sleep
    leftCurState = win32api.GetKeyState(0x01)
    if leftCurState != leftState:
        leftState = leftCurState
        if leftCurState < 0:
            pos = 0
    else:
        if leftCurState < 0:
            pos += 5
##        if leftCurState < 0:
##            #we are pressed, so start identification
##            print("down")
##            analyze()
##    else:
##        if leftCurState < 0:
####            print("hold down")
##            analyze()

    if leftCurState < 0:
        centers = analyze(pos)

    
            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
