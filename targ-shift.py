import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
import glob

max_value = 255
max_type = 4
max_binary_value = 255
min_binary_value = 0

threshold = 'threshold'
angle = 'angle'
votes = 'votes'
minPix = 'min pix'
maxGap = 'max gap'

angle_eps_c = 'angle eps'
dist_eps_c = 'dist eps'
merge_eps_c = 'merge eps'


trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
##trackbar_type2 = 'Type2: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_type2 = 'val'
trackbar_value = 'Value'
trackbar_value2 = 'Value2'
canny_min = 'canny min'
canny_max = 'canny max'
window_name = 'Threshold Demo'
kernel1 = np.ones((1,1), np.uint8)
kernel2 = np.ones((2,2), np.uint8)
kernel3 = np.ones((3,3), np.uint8)
kernelround = np.array([[0,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,0]], dtype=np.uint8)
rot_mask = np.zeros((1,1), np.uint8)
##colors = [(0,0,255), (0,150,255), (0,255,255), (0,255,150), (0,255,0), (150,255,0), (255,255,0), (255,150,0), (255,0,0), (255,0,150), (255,0,255), (255,150,255)]
colors = [(0,0,255)]    #red

def get_spaced_colors(n):
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    
    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

colors.extend(get_spaced_colors(20)[1:])
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
    #0: Binary
    #1: Binary Inverted
    #2: Threshold Truncated
    #3: Threshold to Zero
    #4: Threshold to Zero Inverted
    image_num = cv2.getTrackbarPos(image_value, window_name)

    orig = allIm[image_num]
    src_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    #add a mask on progress
    mask_c = np.ones((orig.shape[0], orig.shape[1]), np.uint8)
##    cv2.circle(mask_c, (145, 145), 125, (255,255,255), thickness=21)
    cv2.circle(mask_c, (145, 145), 125, (0,0,0), thickness=12)
    cv2.circle(mask_c, (145, 145), 50, (0,0,0), thickness=-1)
##    cv2.imshow("circle", mask_c)
##    mask_img = cv2.bitwise_and(orig, orig, mask=mask_c)
    src_gray = cv2.bitwise_and(src_gray, src_gray, mask=mask_c)
##    cv2.imshow("mask", mask_img)

    cv2.imshow("orig", orig)
    cv2.imshow("gray", src_gray)
    '''
    threshold_type = cv2.getTrackbarPos(trackbar_type, window_name)
    threshold_type2 = cv2.getTrackbarPos(trackbar_type2, window_name)
    threshold_value = cv2.getTrackbarPos(trackbar_value, window_name) + 230
    threshold2 = cv2.getTrackbarPos(trackbar_value2, window_name)
    canny_min_val = cv2.getTrackbarPos(canny_min, window_name)
    canny_max_val = cv2.getTrackbarPos(canny_max, window_name)
    '''

    threshold_value = cv2.getTrackbarPos(threshold, window_name) + 230
    angle_value = cv2.getTrackbarPos(angle, window_name)
    votes_value = cv2.getTrackbarPos(votes, window_name)
    minPix_value = cv2.getTrackbarPos(minPix, window_name)
    maxGap_value = cv2.getTrackbarPos(maxGap, window_name)
    
    _, dst = cv2.threshold(src_gray, threshold_value, max_binary_value, 0 )  #250-255, then and with lower increments to enhance down to 230
##    _, dst = cv2.threshold(src_gray, 250, threshold2, threshold_type )
##    for count in range(240, 230, -10):
##        _, temp = cv2.threshold(dst, count, 255, threshold_type )
##        dst = cv2.bitwise_and(
##    _, dst = cv2.threshold(dst, threshold2, min_binary_value, threshold_type2)
    cv2.imshow(window_name, dst)

    dil = cv2.dilate(dst, kernel2, iterations=3)
    cv2.imshow("dil", dil)
    
##    er = cv2.erode(dil, kernelround, iterations=canny_min_val)
##    cv2.imshow("erode", er)

    edges = cv2.Canny(dil, 50, 150, apertureSize=5)
    cv2.imshow("canny", edges)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / angle_value  # angular resolution in radians of the Hough grid
    threshold_votes = votes_value  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = minPix_value  # minimum number of pixels making up a line
    max_line_gap = maxGap_value  # maximum gap in pixels between connectable line segments
    line_image = cv2.cvtColor(dst.copy(), cv2.COLOR_GRAY2RGB) * 0  # creating a blank to draw lines on
    line_image_filtered = line_image.copy()
    line_image_clean = line_image.copy()

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(dil, rho, theta, threshold_votes, np.array([]),
                        min_line_length, max_line_gap)

    origin_x = 144
    origin_y = 143
    
    slope_eps = cv2.getTrackbarPos(angle_eps_c, window_name) #/ 10
    dist_eps = cv2.getTrackbarPos(dist_eps_c, window_name)
    merge_eps = cv2.getTrackbarPos(merge_eps_c, window_name)
    filtered = []
    filtered_angles = []
    added = [0 for x in range(len(lines))]
    radial_im = cv2.cvtColor(src_gray.copy(), cv2.COLOR_GRAY2RGB)
    extended_im_seg = cv2.cvtColor(src_gray.copy(), cv2.COLOR_GRAY2RGB)
    extended_im_orig1 = cv2.cvtColor(src_gray.copy(), cv2.COLOR_GRAY2RGB)
    extended_im_orig2 = cv2.cvtColor(src_gray.copy(), cv2.COLOR_GRAY2RGB)

    houghIm = cv2.cvtColor(dst.copy(), cv2.COLOR_GRAY2RGB)
    houghFiltered = houghIm.copy()
    houghRes1 = []
    lines2 = cv2.HoughLines(dil, rho, np.pi / 180, 25)   #90 for 2 deg
    if lines2 is not None:
        print("found hough")
        for i in range(0, len(lines2)):
            rho = lines2[i][0][0]
            theta = lines2[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

##            pt1a = (int(x0 + 100*(-b)), int(y0 + 100*(a)))
##            pt2a = (int(x0 - 100*(-b)), int(y0 - 100*(a)))
##            cv2.line(houghIm, pt1a, pt2a, (0,0,255), 1, cv2.LINE_AA)
##            cv2.circle(houghIm, pt2a, 5, (0,255,0), 2)
##            cv2.circle(houghIm, (int(x0),int(y0)), 5, (255,0,0), 2)
##
##            print(theta, rho, math.tan(theta))

            if theta != 0:
                slope = -1/math.tan(theta)#-a/b
                inter = y0 - slope * x0
                if abs((origin_x * slope + inter) - origin_y) < merge_eps:
                    print("near origin", slope, rho, theta, math.degrees(theta))
                    cv2.line(houghIm, pt1, pt2, (0,255,0), 1, cv2.LINE_AA)
                    houghRes1.append([theta, rho])
            elif theta == 0 and abs(rho - origin_x) < merge_eps:
                print("vertical near origin")
                cv2.line(houghIm, pt1, pt2, (255,0,0), 1, cv2.LINE_AA)
##                houghRes1.append([theta, rho])
                houghRes1.append([math.pi, rho])
        cv2.circle(houghIm, (origin_x, origin_y), merge_eps, (0,0,255), 1)
    cv2.imshow("hough", houghIm)

    houghRes1.sort()
    '''
    for x in range(len(houghRes1)):
        slope1 = -1/math.tan(houghRes1[x][0])
        inter1 = (math.sin(houghRes1[x][0]) * houghRes1[x][1]) - slope1 * (math.cos(houghRes1[x][0]) * houghRes1[x][1])
        for y in range(x+1, len(houghRes1)):
            #compute intersections
            slope2 = -1/math.tan(houghRes1[y][0])
            inter2 = (math.sin(houghRes1[y][0]) * houghRes1[y][1]) - slope2 * (math.cos(houghRes1[y][0]) * houghRes1[y][1])
            if slope1 == slope2:
                print("parallel")
                continue
            coX = (inter2-inter1)/(slope2-slope1)
            coY = slope1 * coX + inter1
##            print(coX, coY)
            cv2.circle(houghIm, (int(coX),int(coY)), 5, (255,0,0), 1)
    cv2.imshow("inter", houghIm)
    '''
        
    print(len(houghRes1))
    print(houghRes1)

    '''
    preHoughRes = []
    prevBin = 0
    #preprocess- do a wide search: if we get only 2 classes back, then this is a single target
    for x in range(1, len(houghRes1)):
        if abs(math.degrees(houghRes1[x][0]) - math.degrees(houghRes1[x-1][0])) < slope_eps and abs(houghRes1[x][1] - houghRes1[x-1][1]) < dist_eps and x != len(houghRes1)-1:
            continue
        else:
            avgT = 0
            avgR = 0
            count= 0
            z = x
            if x == len(houghRes1)-1:
                if abs(math.degrees(houghRes1[x][0]) - math.degrees(houghRes1[x-1][0])) < slope_eps:
                    print("grouping all")
                    z += 1
                else:
                    print("grouping prev, and binning last")
                    preHoughRes.append([houghRes1[x][0], houghRes1[x][1]])
            for y in range(prevBin, z):
                #average and append to houghRes2
                avgT += houghRes1[y][0]
                avgR += houghRes1[y][1]
                count += 1
            print(count, "in bin")
            preHoughRes.append([avgT/count, avgR/count])
            prevBin = z
    print("Preprocessing done; number of classes:", len(preHoughRes))
    for line in preHoughRes:
        a = math.cos(line[0])
        b = math.sin(line[0])
        x0 = a * line[1]
        y0 = b * line[1]
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(houghFiltered, pt1, pt2, (0,255,0), 1, cv2.LINE_AA)
    cv2.imshow("hough preprocess", houghFiltered)
    '''

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
    print("preprocessing done: number of targets =", numTargets)
    '''
    houghRes2 = []
    cur = math.degrees(houghRes1[-1][0])
    stop = -1
    for x in range(len(houghRes1) - 2, 0, -1):
        if abs(math.degrees(houghRes1[x][0]) - cur) > dist_eps:
            stop = x
            break
    '''
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
                        print("grouping all")
                        z += 1
                    else:
                        print("grouping prev, and binning last")
                        print(houghRes1[x][0], houghRes1[x][1])
                        houghRes2.append([houghRes1[x][0], houghRes1[x][1]])
                for y in range(prevBin, z):
                    #average and append to houghRes2
                    avgT += houghRes1[y][0]
                    avgR += houghRes1[y][1]
                    count += 1
                print(count, "in bin")
                houghRes2.append([avgT/count, avgR/count])
                prevBin = z
    else:
        #classify from stop to end the center
        maxDeg = houghRes1[-1][0]
        maxRad = houghRes1[-1][1]
        minDeg = houghRes1[stop + 1][0]
        minRad = houghRes1[stop + 1][1]
        houghRes2.append([(maxDeg+minDeg)/2, (maxRad+minRad)/2])
    ##    avgT = 0
    ##    avgR = 0
    ##    count = 0
    ##    for x in range(stop, len(houghRes1)):
    ##        avgT += houghRes1[x][0]
    ##        avgR += houghRes1[x][1]
    ##        count += 1
    ##    houghRes2.append([avgT/count, avgR/count])
        
##        prevBin = 0
    ##    for x in range(1,len(houghRes1)):
        for x in range(1, stop + 2):
    ##        print(math.degrees(houghRes1[x][0]), houghRes1[x][1], math.degrees(houghRes1[x-1][0]), houghRes1[x-1][1])
            if abs(math.degrees(houghRes1[x][0]) - math.degrees(houghRes1[x-1][0])) < slope_eps and x != stop + 1: #and x != len(houghRes1)-1:# and abs(houghRes1[x][1] - houghRes1[x-1][1]) < dist_eps:
                continue
            else:
                avgT = 0
                avgR = 0
                count = 0
                z = x
                if x == stop + 1:# len(houghRes1)-1:
                    print('last')
                    if abs(math.degrees(houghRes1[x][0]) - math.degrees(houghRes1[x-1][0])) < slope_eps:
                        print("grouping all")
                        z += 1
                    else:
                        print("grouping prev, and binning last")
                        houghRes2.append([houghRes1[x][0], houghRes1[x][1]])
                for y in range(prevBin, z):
                    #average and append to houghRes2
                    avgT += houghRes1[y][0]
                    avgR += houghRes1[y][1]
                    count += 1
                print(count, "in bin")
                houghRes2.append([avgT/count, avgR/count])
                prevBin = z
        #bin last
    ##    houghRes2.append([houghRes1[stop+1][0], houghRes1[stop+1][1]])
    print()
    print(len(houghRes2))
    '''
    for line in houghRes2:
        a = math.cos(line[0])
        b = math.sin(line[0])
        x0 = a * line[1]
        y0 = b * line[1]
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(houghFiltered, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)
    '''
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
    print(houghRes2)

    #now find center(s) of target(s)
    center = []
    if len(houghRes2) == 2:
        '''
        #take average
        centerT = (houghRes2[0][0] + houghRes2[1][0])/2
        centerR = (houghRes2[0][1] + houghRes2[1][1])/2
        
        if houghRes2[0][0] == 0 or houghRes2[0][1] == 0:
            centerT += math.pi / 2
##            centerR = (-houghRes2[0][1] + houghRes2[1][1]) / 2
##            centerR = abs(houghRes2[0][1] - houghRes2[1][1])
        '''

        '''
        #eric's implementation: only works for vertical
        tempR = math.cos(math.pi - centerT)
        r1 = houghRes2[0][1]/tempR
        r2 = houghRes2[1][1]/tempR
        centerR = -abs(r1-r2)/2
        '''

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
##        radius = 100
##        x = round(origin_x + math.degrees(math.cos(avgAngle)) * radius)
##        y = round(origin_y + math.degrees(math.sin(avgAngle)) * radius)
##        cv2.line(houghFiltered, (origin_x, origin_y), (x,y), (0,255,0), 2)
        
##        center = [centerT, centerR]
        
##        print("center t:", centerT)
    elif len(houghRes2) == 3 or len(houghRes2) == 4:
        #use largest angle
        houghRes2.sort()
##        center = [houghRes2[-1][0], houghRes2[-1][1]]
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
##        a = math.cos(center[0])
##        b = math.sin(center[0])
##        x0 = a * center[1]
##        y0 = b * center[1]
##        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
##        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

        radius = 100
        x = round(origin_x + math.degrees(math.cos(center[0])) * radius)
        y = round(origin_y + math.degrees(math.sin(center[0])) * radius)
        cv2.line(houghFiltered, (origin_x, origin_y), (x,y), (255,0,0), 1, cv2.LINE_AA)
        '''
        print("arctan ver:", math.atan2((y0-1000*a)-origin_y, (x0-1000*(-b))-origin_x))


##        a = math.cos(math.pi/2)
##        b = math.sin(math.pi/2)
##        x0 = a * 100
##        y0 = b * 100
##        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
##        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
##        cv2.line(houghFiltered, pt1, pt2, (0,255,0), 1, cv2.LINE_AA)
##
##        a = math.cos(math.pi)
##        b = math.sin(math.pi)
##        x0 = a * 100
##        y0 = b * 100
##        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
##        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
##        cv2.line(houghFiltered, pt1, pt2, (0,255,0), 1, cv2.LINE_AA)
##
##        a = math.cos(0)
##        b = math.sin(0)
##        x0 = a * 100
##        y0 = b * 100
##        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
##        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
##        cv2.line(houghFiltered, pt1, pt2, (255,0,0), 1, cv2.LINE_AA)

        line1 = [0, 145]
        line2 = [2 * math.pi / 3, 52.5]
        for line in [line1, line2]:
            print("test")
            a = math.cos(line[0])
            b = math.sin(line[0])
            x0 = a * line[1]
            y0 = b * line[1]
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(houghFiltered, pt1, pt2, (0,255,0), 1, cv2.LINE_AA)
        centerT = (2*math.pi/6) + (math.pi / 2)
        centerR = (-145+52.5) / 2
        a = math.cos(centerT)
        b = math.sin(centerT)
        x0 = a * centerR
        y0 = b * centerR
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(houghFiltered, pt1, pt2, (0,255,0), 1, cv2.LINE_AA)
        cv2.circle(houghFiltered, (145,145), 2, (0,255,0), -1)
        cv2.imshow("target result", houghFiltered)
        '''

    '''
    a = math.cos(math.radians(30))
    b = math.sin(math.radians(30))
    x0 = a * 150
    y0 = b * 150
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    cv2.line(houghFiltered, pt1, pt2, (255,0,0), 1, cv2.LINE_AA)
    a = math.cos(math.radians(15))
    b = math.sin(math.radians(15))
    x0 = a * 150
    y0 = b * 150
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    cv2.line(houghFiltered, pt1, pt2, (255,0,0), 1, cv2.LINE_AA)
    '''
    
    cv2.imshow("hough filtered", houghFiltered)
    print("done with hough")
##        if theta != 0:
##            slope = -1/math.tan(theta)#-a/b
##            inter = y0 - slope * x0
##            if abs((origin_x * slope + inter) - origin_y) < merge_eps:
##                print("near origin", slope, rho, theta, math.degrees(theta))
##                cv2.line(houghIm, pt1, pt2, (0,255,0), 1, cv2.LINE_AA)
##                houghRes1.append([theta, rho])
##        elif theta == 0 and abs(rho - origin_x) < merge_eps:
##            print("vertical near origin")
##            cv2.line(houghIm, pt1, pt2, (255,0,0), 1, cv2.LINE_AA)

##    print(len(lines))
    print()
    for line in lines:
##        print(line)
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),1)
    print()
        # Draw the lines on the  image
##    lines_edges = cv2.addWeighted(dst, 0.8, line_image, 1, 0)
    cv2.imshow("lines", line_image)

    #merge on proximity and slope
    '''
    slope_eps = 0.5
    dist_eps = 7
    '''

    """

    for x in range(len(lines)):     #order X1,Y1 to be closest to origin
        x1,y1,x2,y2 = lines[x][0][0:4]
        if math.sqrt(math.pow(x1-origin_x, 2) + math.pow(y1-origin_y, 2)) > math.sqrt(math.pow(x2-origin_x, 2) + math.pow(y2-origin_y, 2)):
            lines[x][0][0:2] = [x2,y2]
            lines[x][0][2:4] = [x1,y1]

    for x in range(len(lines)):
        if added[x] == 1:
            continue
        xa1,ya1,xa2,ya2 = lines[x][0][0:4]
        #if slope from origin to (xa1, ya1) is similar to slope of line, then this is a radial line
        angleOrigin = math.degrees(math.atan2(ya1-origin_y, xa1-origin_x))
        angleLine = math.degrees(math.atan2(ya2-ya1, xa2-xa1))
        added[x] = 1
        if abs(angleOrigin - angleLine) < slope_eps:
##            cv2.line(radial_im, (xa1,ya1), (xa2,ya2), colors[0], 2)
            #find other lines similar in slope to this, and take average
##            for y in range(x, len(lines)):
##                if added[y] == 1:
##                    continue
##                xb1,yb1,xb2,yb2 = lines[y][0][0:4]
##                angleSecondary = math.degrees(math.atan2(yb2-yb1, xb2-xb1))
##                if abs(angleLine - angleSecondary) < slope_eps:
##                    xa1 = (xa1 + xb1) / 2
##                    ya1 = (ya1 + yb1) / 2
##                    xa2 = (xa2 + xb2) / 2
##                    ya2 = (ya2 + yb2) / 2
##                    added[y] = 1

            #find other lines similar in slope to current and origin
            for y in range(x, len(lines)):
                if added[y] == 1:
                    continue
                xb1,yb1,xb2,yb2 = lines[y][0][0:4]
                angleCur = math.degrees(math.atan2(ya2-ya1, xa2-xa1))   #angle of current estimate
                angleOriginCur1 = math.degrees(math.atan2(ya1-origin_y, xa1-origin_x))  #angle of current estimate pt1 to origin
                angleOriginCur2 = math.degrees(math.atan2(ya2-origin_y, xa2-origin_x))  #angle of current estimate pt2 to origin
                angleSecondary = math.degrees(math.atan2(yb2-yb1, xb2-xb1))     #angle of new segment
                angleOriginSec1 = math.degrees(math.atan2(yb1-origin_y, xb1-origin_x))  #angle of new seg pt1 to origin
                angleOriginSec2 = math.degrees(math.atan2(yb2-origin_y, xb2-origin_x))  #angle of new seg pt2 to origin
                if abs(angleLine - angleSecondary) < dist_eps and abs(angleOriginCur1 - angleOriginSec1) < slope_eps and abs(angleOriginCur2 - angleOriginSec2) < slope_eps:
                    xa1 = (xa1 + xb1) / 2
                    ya1 = (ya1 + yb1) / 2
                    xa2 = (xa2 + xb2) / 2
                    ya2 = (ya2 + yb2) / 2
                    added[y] = 1
                elif abs(angleLine - angleSecondary) < dist_eps or abs(angleOriginCur1 - angleOriginSec1) < slope_eps or abs(angleOriginCur2 - angleOriginSec2) < slope_eps:
                    added[y] = 1
                    #discard this
            filtered.append([int(xa1), int(ya1), int(xa2), int(ya2)])
##            filtered_angles.append((360 - math.degrees(math.atan2(ya2 - origin_y, xa2 - origin_x))) % 360 - 90)
            filtered_angles.append(math.atan2(ya2 - origin_y, xa2 - origin_x))
    print(len(filtered))
    for line in filtered:
        x1,y1,x2,y2 = line[0:4]
        cv2.line(radial_im, (origin_x, origin_y), (x1, y1), (0,0,255), 1)
        cv2.line(radial_im, (x1, y1), (x2, y2), (255,0,0), 1)

        #extend segments past origin and segment
        slope = (y2-y1)/(x2-x1)
        b = round(y1 - (slope * x1))
        cv2.line(extended_im_seg, (0, b), (x2, y2), (0,0,255), 1)

        slope = (y1-origin_y)/(x1-origin_x)
        b = round(y1 - slope * x1)
        cv2.line(extended_im_orig1, (0, b), (x2, y2), (0,0,255), 1)

        slope = (y2-origin_y)/(x2-origin_x)
        b = round(y2 - slope * x2)
        cv2.line(extended_im_orig2, (0, b), (x2, y2), (0,0,255), 1)

    cv2.imshow("radial lines", radial_im)
##    cv2.imshow("extended seg", extended_im_seg)
##    cv2.imshow("extended orig1", extended_im_orig1)
    cv2.imshow("extended orig2", extended_im_orig2)
    '''
    centers = []    #take center based off origin and pt2
    if len(filtered) == 2:
        xa1,ya1,xa2,ya2 = filtered[0]
        xb1,yb1,xb2,yb2 = filtered[1]
        centers.append([int((xa2 + xb2) / 2), int((ya2 + yb2) / 2)])
    elif len(filtered) == 4:
        xa1,ya1,xa2,ya2 = filtered[0]
        xb1,yb1,xb2,yb2 = filtered[1]
        xc1,yc1,xc2,yc2 = filtered[2]
        xd1,yd1,xd2,yd2 = filtered[3]
        angleCur = math.degrees(math.atan2(ya2-ya1, xa2-xa1))   #angle of current estimate
        for line in filtered:
            print(line, math.degrees(math.atan2(line[3]-line[1], line[2]-line[0])))
    else:
        print("bad class")
    for pt in centers:
        print(pt)
        cv2.line(radial_im, (origin_x, origin_y), (pt[0],pt[1]), (0,255,0), 2)
    cv2.imshow("centers", radial_im)
    '''
    centers = []
    filtered_angles.sort()
    print(filtered_angles)
    if len(filtered_angles) == 2:
        centers.append((filtered_angles[0] + filtered_angles[1]) / 2)
    elif len(filtered_angles) == 4:
        centers.append((filtered_angles[0] + filtered_angles[1]) / 2)
        centers.append((filtered_angles[2] + filtered_angles[3]) / 2)
    else:
        print("bad class")
    radius = 50
    for ang in centers:
        print(ang)
        '''
        print(math.degrees(math.cos(math.radians(ang))))
        print(math.degrees(math.sin(math.radians(ang))))
        x = round(origin_x + math.degrees(math.cos(math.radians(ang + 90))) * radius)
        y = round(origin_y - math.degrees(math.sin(math.radians(ang + 90))) * radius)
        '''
        x = round(origin_x + math.degrees(math.cos(ang)) * radius)
        y = round(origin_y + math.degrees(math.sin(ang)) * radius)
        cv2.line(radial_im, (origin_x, origin_y), (x,y), (0,255,0), 2)
    cv2.imshow("centers", radial_im)

    """
    
    """
    for x in range(len(lines)):     #order X L->R
        x1,y1,x2,y2 = lines[x][0][0:4]
        if x1 > x2:
            lines[x][0][0:4] = x2, y2, x1, y1
##        elif x1 == x2:
##            lines[x][0][0] -= 1
    seg = 0
    for x in range(len(lines)):
        if added[x] == 1:
            continue
        added[x] = 1
        xa1,ya1,xa2,ya2 = lines[x][0][0:4]
        #diagonistic canvas
        line_image_temp = line_image_clean.copy()
        cv2.line(line_image_temp, (xa1,ya1), (xa2, ya2), colors[0], 2)
        cur = 1
        seg += 1
        print()
        print("segment", seg, xa1,ya1,xa2,ya2)
        for y in range(len(lines)):
            if added[y] == 1:
                continue
            xb1,yb1,xb2,yb2 = lines[y][0][0:4]
            '''
            if xa2-xa1 == 0 or xb2-xb1 == 0:    #vertical, so use dx/dy
                slope1 = (xa2-xa1)/(ya2-ya1)
                slope2 = (xb2-xb1)/(yb2-yb1)
            else:
                slope1 = (ya2-ya1)/(xa2-xa1)
                slope2 = (yb2-yb1)/(xb2-xb1)
            dist1 = math.sqrt(math.pow(xa1-xb1, 2) + math.pow(ya1-yb1, 2))  #distance between first endpoints
            dist2 = math.sqrt(math.pow(xa2-xb2, 2) + math.pow(ya2-yb2, 2))  #distance between second endpoints
            if abs(slope1 - slope2) < slope_eps and (dist1 < dist_eps or dist2 < dist_eps):
                added[y] = 1
                if dist1 < dist_eps and dist2 < dist_eps:
                    #these lines are close together, so take average
                    xa1 = (xa1 + xb1) / 2
                    ya1 = (ya1 + yb1) / 2
                    xa2 = (xa2 + xb2) / 2
                    ya2 = (ya2 + yb2) / 2
                else:
                    #these lines have similar slope, but one is longer than the other -> take longer
                    if math.sqrt(math.pow(xa1-xa2, 2) + math.pow(ya1-ya2, 2)) < math.sqrt(math.pow(xb1-xb2, 2) + math.pow(yb1-yb2, 2)):
                        xa1 = xb1
                        ya1 = yb1
                        xa2 = xb2
                        ya2 = yb2
            '''
            dist1 = math.sqrt(math.pow(xa1-xb1, 2) + math.pow(ya1-yb1, 2))  #distance between first endpoints
            dist2 = math.sqrt(math.pow(xa2-xb2, 2) + math.pow(ya2-yb2, 2))  #distance between second endpoints
            dist3 = math.sqrt(math.pow(xb1-xa2, 2) + math.pow(yb1-ya2, 2))  #distance for merging two lines, a on left
            dist4 = math.sqrt(math.pow(xa1-xb2, 2) + math.pow(ya1-yb2, 2))  #distance for merging two lines, b on left
            angle1 = math.degrees(math.atan2(ya2-ya1, xa2-xa1))
            angle2 = math.degrees(math.atan2(yb2-yb1, xb2-xb1))
            angle3 = math.degrees(math.atan2(xa2-xa1, ya2-ya1))
            angle4 = math.degrees(math.atan2(xb2-xb1, yb2-yb1))
##            if xa2-a1 == 0:
##                angle1 = math.degrees
            if abs(angle1 - angle2) < slope_eps:# or ((abs(angle1) == 90 or abs(angle2) == 90) and abs(abs(angle1) - abs(angle2)) < slope_eps): #similar slope
                cv2.line(line_image_temp, (xb1,yb1), (xb2,yb2), colors[cur], 2)
##                cv2.imshow("seg" + str(seg) + " current" + str(cur), line_image_temp)
                if dist1 < dist_eps and dist2 < dist_eps:   #similar line segments
                    print(cur, "left similar segs", [xa1,ya1,xa2,ya2], angle1, [xb1,yb1,xb2,yb2], angle2)
                    #take average
                    xa1 = (xa1 + xb1) / 2
                    ya1 = (ya1 + yb1) / 2
                    xa2 = (xa2 + xb2) / 2
                    ya2 = (ya2 + yb2) / 2
                    added[y] = 1
                elif dist1 < dist_eps or dist2 < dist_eps:  #only one of the endpoints are close
                    print(cur, "left similar ep", [xa1,ya1,xa2,ya2], angle1, [xb1,yb1,xb2,yb2], angle2)
##                    if abs(angle1) == 90:
##                        if dist1 < dist_eps:
##                            #get angle between, if closer to vertical then merge lines; otherwise take average
                    #take longer
                    if math.sqrt(math.pow(xa1-xa2, 2) + math.pow(ya1-ya2, 2)) < math.sqrt(math.pow(xb1-xb2, 2) + math.pow(yb1-yb2, 2)):
                        xa1 = xb1
                        ya1 = yb1
                        xa2 = xb2
                        ya2 = yb2
                    added[y] = 1
                elif dist3 < merge_eps:
                    print(cur, "left merge left a->b", [xa1,ya1,xa2,ya2], angle1, [xb1,yb1,xb2,yb2], angle2)
##                    if abs(angle1) == 90:
##                        
                    #merge lines
                    xa2 = xb2
                    ya2 = yb2
                    added[y] = 1
                elif dist4 < merge_eps:
                    print(cur, "left merge right b->a", [xa1,ya1,xa2,ya2], angle1, [xb1,yb1,xb2,yb2], angle2)
                    xa1 = xb1
                    ya1 = yb1
                    added[y] = 1
                else:
                    print(cur, "left no merge", [xa1,ya1,xa2,ya2], angle1, [xb1,yb1,xb2,yb2], angle2, dist1, dist2, dist3)
                    print("orthogonal", angle3, angle4)
                
                cur += 1
            elif abs(angle1 - angle2) > 180 - slope_eps:
                if dist1 < dist_eps and dist2 < dist_eps:   #similar line segments
                    print(cur, "right similar segs", [xa1,ya1,xa2,ya2], angle1, [xb1,yb1,xb2,yb2], angle2)
                    #take average
                    xa1 = (xa1 + xb1) / 2
                    ya1 = (ya1 + yb1) / 2
                    xa2 = (xa2 + xb2) / 2
                    ya2 = (ya2 + yb2) / 2
                    added[y] = 1
                elif dist1 < merge_eps:
                    #merge
                    print(cur, "right merge left a->b", [xa1,ya1,xa2,ya2], angle1, [xb1,yb1,xb2,yb2], angle2)
                    xa1 = xb2
                    ya1 = yb2
                    added[y] = 1
                elif dist2 < merge_eps:
                    print(cur, "right merge right b->a", [xa1,ya1,xa2,ya2], angle1, [xb1,yb1,xb2,yb2], angle2)
                    xa2 = xb1
                    ya2 = yb1
                    added[y] = 1
                elif dist3 < dist_eps or dist4 < dist_eps:
                    #take longer
                    if math.sqrt(math.pow(xa1-xa2, 2) + math.pow(ya1-ya2, 2)) < math.sqrt(math.pow(xb1-xb2, 2) + math.pow(yb1-yb2, 2)):
                        xa1 = xb1
                        ya1 = yb1
                        xa2 = xb2
                        ya2 = yb2
                    added[y] = 1
                else:
                    print(cur, "right no merge", [xa1,ya1,xa2,ya2], angle1, [xb1,yb1,xb2,yb2], angle2, dist1, dist2, dist3)
                    print("orthogonal", angle3, angle4)
                '''
                elif dist1 < dist_eps or dist2 < dist_eps:  #only one of the endpoints are close
                    print(cur, "right similar ep", [xa1,ya1,xa2,ya2], angle1, [xb1,yb1,xb2,yb2], angle2)
##                    if abs(angle1) == 90:
##                        if dist1 < dist_eps:
##                            #get angle between, if closer to vertical then merge lines; otherwise take average
                    #take longer
                    if math.sqrt(math.pow(xa1-xa2, 2) + math.pow(ya1-ya2, 2)) < math.sqrt(math.pow(xb1-xb2, 2) + math.pow(yb1-yb2, 2)):
                        xa1 = xb1
                        ya1 = yb1
                        xa2 = xb2
                        ya2 = yb2
                    added[y] = 1
                elif dist3 < merge_eps:
                    print(cur, "right merge left a->b", [xa1,ya1,xa2,ya2], angle1, [xb1,yb1,xb2,yb2], angle2)
##                    if abs(angle1) == 90:
##                        
                    #merge lines
                    xa2 = xb2
                    ya2 = yb2
                    added[y] = 1
                elif dist4 < merge_eps:
                    print(cur, "right merge right b->a", [xa1,ya1,xa2,ya2], angle1, [xb1,yb1,xb2,yb2], angle2)
                    xa1 = xb1
                    ya1 = yb1
                    added[y] = 1
                '''
                cur += 1
        cv2.imshow("seg" + str(seg) + " current" + str(cur), line_image_temp)
        '''
        if math.sqrt(math.pow(xa1-xb1, 2) + math.pow(ya1-yb1, 2)) < dist_eps and abs(angle1 - angle2) < slope_eps:
            added[y] = 1
            #these lines have similar slope, but one is longer than the other -> take longer
            if math.sqrt(math.pow(xa1-xa2, 2) + math.pow(ya1-ya2, 2)) < math.sqrt(math.pow(xb1-xb2, 2) + math.pow(yb1-yb2, 2)):
                xa1 = xb1
                ya1 = yb1
                xa2 = xb2
                ya2 = yb2
        '''
                
        filtered.append([int(xa1),int(ya1),int(xa2),int(ya2)])
    
    print(len(filtered))
    print(filtered)
    #display
    for line,col in zip(filtered,colors):
        x1,y1,x2,y2 = line[0:4]
        cv2.line(line_image_filtered, (x1,y1), (x2,y2), col, 1)
        cv2.imshow("filtered line", line_image_filtered)
    """

'''
    #for each line in lines: find all neighboring lines and add to a set
    #if a line has already been added, skip
    eps = 11   #radius
    #order lines
    for x in range(len(lines)):
        x1,y1,x2,y2 = lines[x][0][0:4]
        if x1 > x2:
            lines[x][0][0:4] = x2, y2, x1, y1

    filtered_lines = []
    filtered_lines_int = []
    count = 0
    added = [0 for x in range(len(lines))]
    for x in range(len(lines)):
        if added[x] == 1:
            continue
        added[x] = 1
##        temp_lines = lines[x][0]
        xa1,ya1,xa2,ya2 = lines[x][0][0:4]
##        print(type(temp_lines))
        for y in range(x, len(lines)):
            if added[y] == 1:
                continue
            #this assumes that each element in the lines array contains only one subarray
##            xa1,ya1,xa2,ya2 = temp_lines[0:4]
            xb1,yb1,xb2,yb2 = lines[y][0][0:4]
            if math.sqrt(math.pow(xa1-xb1, 2) + math.pow(ya1-yb1, 2)) < eps and math.sqrt(math.pow(xa2-xb2, 2) + math.pow(ya2-yb2, 2)) < eps:
                #add it to a set, reaverage, and mark y as completed
                xa1 = (xa1 + xb1) / 2
                ya1 = (ya1 + yb1) / 2
                xa2 = (xa2 + xb2) / 2
                ya2 = (ya2 + yb2) / 2
##                temp_lines = [xa1, ya1, xa2, ya2]
##                print(type(temp_lines))
##                print(temp_lines)
                added[y] = 1
##        print(type(temp_lines))
##        filtered_lines.append(temp_lines)
        filtered_lines.append([xa1,ya1,xa2,ya2])
        filtered_lines_int.append([int(xa1),int(ya1),int(xa2),int(ya2)])
    print(len(filtered_lines))
    print(filtered_lines)

    for line,col in zip(filtered_lines_int,colors):
        x1,y1,x2,y2 = line[0:4]
        cv2.line(line_image_filtered, (x1,y1), (x2,y2), col, 1)
        cv2.imshow("filtered line", line_image_filtered)

    #merge on slopes
    slope_eps = 0.6
    dist_eps = 3
    clean_lines = []
    added = [0 for x in range(len(filtered_lines))]
    for x in range(len(filtered_lines)):
        if added[x] == 1:
            continue
        added[x] = 1
        xa1,ya1,xa2,ya2 = filtered_lines[x][0:4]
        for y in range(x, len(filtered_lines)):
            if added[y] == 1:
                continue
            xb1,yb1,xb2,yb2 = filtered_lines[y][0:4]
            #similar slope and have an endpoint that is near
            if xa2-xa1 == 0 or xb2-xb1 == 0:    #vertical, so use dx/dy
                slope1 = (xa2-xa1)/(ya2-ya1)
                slope2 = (xb2-xb1)/(yb2-yb1)
            else:
                slope1 = (ya2-ya1)/(xa2-xa1)
                slope2 = (yb2-yb1)/(xb2-xb1)
##            print(slope1, slope2, math.sqrt(math.pow(xa1-xb1, 2) + math.pow(ya1-yb1, 2)), math.sqrt(math.pow(xa2-xb2, 2) + math.pow(ya2-yb2, 2)))
            if abs(slope1 - slope2) < slope_eps and (math.sqrt(math.pow(xa1-xb1, 2) + math.pow(ya1-yb1, 2)) < eps or math.sqrt(math.pow(xa2-xb2, 2) + math.pow(ya2-yb2, 2)) < eps):
                #take longer line
                if math.sqrt(math.pow(xa1-xa2, 2) + math.pow(ya1-ya2, 2)) < math.sqrt(math.pow(xb1-xb2, 2) + math.pow(yb1-yb2, 2)):
                    xa1 = xb1
                    ya1 = yb1
                    xa2 = xb2
                    ya2 = yb2
                added[y] = 1
        clean_lines.append([int(xa1),int(ya1),int(xa2),int(ya2)])
    print(len(clean_lines))
    print(clean_lines)

    for line,col in zip(clean_lines, colors):
        x1,y1,x2,y2 = line[0:4]
        cv2.line(line_image_clean, (x1,y1), (x2,y2), col, 2)
        cv2.imshow("clean line", line_image_clean)
'''

'''
    #corners
    crn = cv2.cornerHarris(dil,2,15,0.15)
    #result is dilated for marking the corners, not important
    crn = cv2.dilate(crn,None)
    # Threshold for an optimal value, it may vary depending on the image.
    crnimg = cv2.cvtColor(dst.copy(), cv2.COLOR_GRAY2RGB)
    crnimg[crn>0.01*crn.max()]=[0,0,255]

    cv2.imshow('crn',crnimg)
'''
    
##    threshold_type = cv2.getTrackbarPos(trackbar_type, window_name)
##    threshold_type2 = cv2.getTrackbarPos(trackbar_type2, window_name)
##    threshold_value = cv2.getTrackbarPos(trackbar_value, window_name)
##    threshold2 = cv2.getTrackbarPos(trackbar_value2, window_name)
##    _, dst = cv2.threshold(src_gray, threshold_value, max_binary_value, threshold_type )
##    _, dst = cv2.threshold(dst, threshold2, min_binary_value, threshold_type2)
##    cv2.imshow(window_name, dst)
##    blur = cv2.GaussianBlur(dst, (19,19), 0)
##    cv2.imshow("blur", blur)
##
####    morph = cv2.dilate(dst, rot_mask, iterations=1)
##    morph = cv2.erode(dst, kernel, iterations=1)
####    morph = cv2.dilate(morph, kernel, iterations=2)
####    morph = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)
##    cv2.imshow("morph1", morph)
##
##    flood = morph.copy()
##    floodmask = np.zeros((dst.shape[0]+2, dst.shape[1]+2), np.uint8)
##    cv2.floodFill(flood, floodmask, (0,0), 255)
##    flood = cv2.bitwise_not(flood)
##    floodout = morph | flood
##
##    floodout = cv2.erode(floodout, kernel, iterations=1)
##    floodout = cv2.dilate(floodout, kernel, iterations=2)
##    cv2.imshow("flood", floodout)
##
##    floodcont = cv2.cvtColor(floodout.copy(), cv2.COLOR_GRAY2RGB)
##    im2, contours, hiearchy = cv2.findContours(floodout, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
##    print("image", image_num)
##    print(contours)
##    cv2.drawContours(floodcont, contours, -1, (0,0,255), 2)
##    cv2.imshow("cont", floodcont)
##
##    floodctr = floodcont.copy()
##    avgCtr = np.zeros((len(contours), 2), np.float64)
##    index = 0
##    print()
##    for contour in contours:
####        print(contour)
####        print(contour[:,0,0], np.mean(contour[:,0,0]))
####        print(contour[:,0,1], np.mean(contour[:,0,1]))
##        print("avg")
##        avgX = np.mean(contour[:,0,0])
##        avgY = np.mean(contour[:,0,1])
##        print(avgX, avgY)
##        avgCtr[index] = np.array([avgX, avgY])
##        floodctr[int(round(avgY)), int(round(avgX))] = (0,255,0)
##        print(avgCtr[index])
##        index += 1
##    print("all")
##    print(avgCtr)
####    cv2.drawContours(floodctr, avgCtr, -1, (0, 255, 0), 2)
##    
##    
##    cv2.imshow("ctr", floodctr)
## [Threshold_Demo]


## [window]
# Create a window to display results
cv2.namedWindow(window_name)
## [window]

## [trackbar]
cv2.createTrackbar(image_value, window_name, 16, image_val_max, Threshold_Demo)

cv2.createTrackbar(threshold, window_name, 7, 25, Threshold_Demo)  #230 + 0-25
cv2.createTrackbar(angle, window_name, 180, 360, Threshold_Demo) #95
cv2.createTrackbar(votes, window_name, 8, 25, Threshold_Demo)
cv2.createTrackbar(minPix, window_name, 5, 30, Threshold_Demo)
cv2.createTrackbar(maxGap, window_name, 5, 25, Threshold_Demo)

cv2.createTrackbar(angle_eps_c, window_name, 2, 45, Threshold_Demo)    #39 for left join only
cv2.createTrackbar(dist_eps_c, window_name, 11, 45, Threshold_Demo)
cv2.createTrackbar(merge_eps_c, window_name, 8, 20, Threshold_Demo)
'''
# Create Trackbar to choose type of Threshold
cv2.createTrackbar(trackbar_type, window_name , 0, max_type, Threshold_Demo)
cv2.createTrackbar(trackbar_type2, window_name ,5, 30, Threshold_Demo)
# Create Trackbar to choose 1st Threshold value
cv2.createTrackbar(trackbar_value, window_name , 12, 25, Threshold_Demo)    #0-25 for 230-255
cv2.createTrackbar(trackbar_value2, window_name , 95, 360, Threshold_Demo)   #190

cv2.createTrackbar(canny_min, window_name , 8, 25, Threshold_Demo)
cv2.createTrackbar(canny_max, window_name , 5, 25, Threshold_Demo)
## [trackbar]
'''

# Call the function to initialize
Threshold_Demo(0)
# Wait until user finishes program
cv2.waitKey()
