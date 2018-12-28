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

##current_mouse_time = time.time()
##timeTick = 5/72    #number of seconds per tick

all_mask = cv2.imread("./filled/20181226213622_1.jpg")[305:605, 655:955]
mask_w = 18
mask_h = 3.2
mask_r = 117
mask_cx = 144
mask_cy = 145
mask_y = 2 * math.degrees(math.asin((mask_h/2)/mask_r))
mask_x = (5 - mask_y) / 2
mask_z = (mask_y / 2) + mask_x
zero_mask = np.zeros((all_mask.shape[0], all_mask.shape[1]), np.uint8)

mask_eps = 8
mask_matchLim = 0.11

#create the mask for progress analysis
allMasks = []
mainMask = []
boundingPts = []
for deg in range(0, 180, 5):
    if deg > 90:
        offset = 0
    else:
        offset = 1
    curD = (270 + deg + offset)
    if curD >= 360: curD -= 360

    rot_mask = zero_mask

    Ax = mask_cx + mask_r * math.cos(math.radians(mask_x + curD))
    Ay = mask_cy + mask_r * math.sin(math.radians(mask_x + curD))
    Bx = mask_cx + mask_r * math.cos(math.radians(mask_x + mask_y + curD))
    By = mask_cy + mask_r * math.sin(math.radians(mask_x + mask_y + curD))

    Cx = Ax + mask_w * math.cos(math.radians(mask_z + curD))
    Cy = Ay + mask_w * math.sin(math.radians(mask_z + curD))
    Dx = Bx + mask_w * math.cos(math.radians(mask_z + curD))
    Dy = By + mask_w * math.sin(math.radians(mask_z + curD))

    pts = np.array([[Ax,Ay], [Bx,By], [Cx, Cy], [Dx, Dy]], np.int32)
    rect= cv2.minAreaRect(pts)
    rot = cv2.boxPoints(rect)
    rot = np.int0(rot)

    cv2.drawContours(rot_mask, [rot], 0, (255,255,255), -1)

    allMasks.append(rot_mask)

    mainMask.append(cv2.bitwise_and(all_mask, all_mask, mask=rot_mask))

    b_x,b_y,b_w,b_h = cv2.boundingRect(pts)
    boundingPts.append([b_x,b_y,b_w,b_h])

#main function to analyze targets and progress
def analyze():
    originalImg = np.array(ImageGrab.grab(bbox=(655, 305, 955, 605)))
    img = cv2.cvtColor(originalImg, cv2.COLOR_BGR2GRAY)
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
##        print("no mining detected")
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
##        print("bad classification")
        return

##    #draw current position
##    clock_offset = 5
##    diffTime = time.time() - current_mouse_time
##    print(diffTime/timeTick)
##    diffTime = round(diffTime/timeTick) * 5.95
##    x = round(origin_x + math.degrees(math.cos(math.radians(diffTime + clock_offset + 90))) * 100)
##    y = round(origin_x + math.degrees(math.sin(math.radians(diffTime + clock_offset + 90))) * 100)
##    cv2.line(houghFiltered, (origin_x, origin_y), (x,y), (0,255,0), 1, cv2.LINE_AA)

    originalImg = np.array(ImageGrab.grab(bbox=(655, 305, 955, 605)))[...,::-1]
    cv2.imshow("orig", originalImg)

    #find current position
    ''''''
    posImg = originalImg.copy()
    prevMiss = 0
    for deg in range(0, 180, 5):
        linIdx = int(deg/5)
        if deg > 90:
            offset = 0
        else:
            offset = 1
        curD = (270 + deg + offset)
        if curD >= 360: curD -= 360

        b_x,b_y,b_w,b_h = boundingPts[linIdx]

        curAll = mainMask[linIdx]
        curTest = cv2.bitwise_and(originalImg, originalImg, mask=allMasks[linIdx])

##        cv2.imshow("mask img", curAll)
##        cv2.imshow("cur mask", curTest)

        template = curAll[b_y:b_y+b_h, b_x:b_x+b_w]
        temp_seg = curTest[b_y:b_y+b_h, b_x:b_x+b_w]

##        curAll = all_mask[b_y:b_y+b_h, b_x:b_x+b_w]
##        curTest = originalImg[b_y:b_y+b_h, b_x:b_x+b_w]
##
##        template = cv2.bitwise_and(curAll, curAll, mask=rot_mask)
##        temp_seg = cv2.bitwise_and(curTest, curTest, mask=rot_mask)
        
        tempArr = template.flatten()
        idx = tempArr.nonzero()
        templateT = tempArr[idx].astype(np.int16)
        segT = temp_seg.flatten()[idx].astype(np.int16)
        diff = np.absolute(templateT - segT)
        count = len(diff)
        acc = len(np.where(diff < mask_eps)[0])
        
        if acc/count > mask_matchLim:
            prevMiss = 0
            #draw line
            x = round(origin_x + math.degrees(math.cos(math.radians(curD + 2.5))) * 200)
            y = round(origin_y + math.degrees(math.sin(math.radians(curD + 2.5))) * 200)
            cv2.line(posImg, (origin_x, origin_y), (x,y), (0,255,0), 1, cv2.LINE_AA)
        else:
            if prevMiss > 0:
                break
            prevMiss += 1
##            x = round(origin_x + math.degrees(math.cos(math.radians(curD + 2.5))) * 200)
##            y = round(origin_y + math.degrees(math.sin(math.radians(curD + 2.5))) * 200)
##            cv2.line(posImg, (origin_x, origin_y), (x,y), (255,0,0), 1, cv2.LINE_AA)
    cv2.imshow("pos img", posImg)
    ''''''

    if len(center) != 0:
        radius = 100
        x = round(origin_x + math.degrees(math.cos(center[0])) * radius)
        y = round(origin_y + math.degrees(math.sin(center[0])) * radius)
##        cv2.line(houghFiltered, (origin_x, origin_y), (x,y), (255,0,0), 1, cv2.LINE_AA)
        cv2.line(houghFiltered, (origin_x, origin_y), (x,y), (0,0,255), 1, cv2.LINE_AA)
    
    cv2.imshow("hough filtered", houghFiltered)

##    print()

    return center

while (True):
    #if key pressed start loooking, otherwise sleep
    leftCurState = win32api.GetKeyState(0x01)
    if leftCurState != leftState:
        leftState = leftCurState
        if leftCurState < 0:
            print("down")
            current_mouse_time = time.time()
        else:
            print("up")

    if leftCurState < 0:
        analyze()
##        centers = analyze()

    
            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
