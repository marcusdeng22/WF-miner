import cv2
import math
import numpy as np
import glob
import time

all_mask = cv2.imread("./filled/20181226213622_1.jpg")[305:605, 655:955]
cv2.imshow("mask orig", all_mask)

##test_img = cv2.imread("./filled/20181226213939_1.jpg")[305:605, 655:955]    #double part almost full
##cv2.imshow("test orig", test_img)

window_name = 'Progress Template'

allIm = []
count = 0
for img in glob.glob("./new-train2/*.jpg"):
##for img in glob.glob("./double-train/*.jpg"):
    count += 1
    n = cv2.imread(img)[305:605, 655: 955]
    allIm.append(n)
##    if count > 50: break
orig = allIm[0]
image_value = "image"
image_val_max = len(allIm) - 1

tempMeth = 'method'
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

def match(val):
    image_num = cv2.getTrackbarPos(image_value, window_name)
    test_img = allIm[image_num]
    cv2.imshow("test img", test_img)
    
    w = 18
##    h = 4
    h = 3.2
    r = 117
    cx = 144
    cy = 145
    y = 2 * math.degrees(math.asin((h/2)/r))
    x = (5 - y) / 2
    z = (y / 2) + x
##    x += 1
##    y -= 1
    rot_mask = np.zeros((all_mask.shape[0], all_mask.shape[1]), np.uint8)

    temp = test_img.copy()

    for deg in range(0, 360, 5):
        Ax = cx + r * math.cos(math.radians(x + deg))
        Ay = cy + r * math.sin(math.radians(x + deg))
        Bx = cx + r * math.cos(math.radians(x + y + deg))
        By = cy + r * math.sin(math.radians(x + y + deg))

        Cx = Ax + w * math.cos(math.radians(z + deg))
        Cy = Ay + w * math.sin(math.radians(z + deg))
        Dx = Bx + w * math.cos(math.radians(z + deg))
        Dy = By + w * math.sin(math.radians(z + deg))

        cv2.line(temp, (cx,cy), (int(Ax), int(Ay)), (0,0,255), 1)
        cv2.line(temp, (cx,cy), (int(Bx), int(By)), (0,255,0), 1)
        cv2.line(temp, (int(Ax), int(Ay)), (int(Cx), int(Cy)), (0,0,255), 1)
        cv2.line(temp, (int(Bx), int(By)), (int(Dx), int(Dy)), (0,255,0), 1)

        pts = np.array([[Ax,Ay], [Bx,By], [Cx, Cy], [Dx, Dy]], np.int32)
        rect= cv2.minAreaRect(pts)
        rot = cv2.boxPoints(rect)
        rot = np.int0(rot)
##        if deg == 0: print(rot)
        cv2.drawContours(rot_mask, [rot], 0, (255,255,255), -1)
    all_mask_filt = cv2.bitwise_and(all_mask, all_mask, mask=rot_mask)
    cv2.imshow("mask_rot", all_mask_filt)

    test_mask_filt = cv2.bitwise_and(test_img, test_img, mask=rot_mask)
    cv2.imshow("masked test", test_mask_filt)

    cv2.imshow("temp", temp)

    #alternate: start at upper middle and compute the mask on both images and compare
    #then once hit, compute masks simultaneously and compare (do a mask and compare in same iter, rather than after)

    temp2 = test_img.copy()
    for deg in range(0, 180, 5):
        if deg > 90:
            offset = 0
        else:
            offset = 1
        curD = (270 + deg + offset)
        if curD >= 360: curD -= 360
##        print(curD, math.radians(curD))
##        tx = cx + 1000 * math.cos(math.radians(curD))
##        ty = cy + 1000 * math.sin(math.radians(curD))
##        cv2.line(temp, (cx,cy), (int(tx),int(ty)), (0,255,0), 1)
        Ax = cx + r * math.cos(math.radians(x + curD))
        Ay = cy + r * math.sin(math.radians(x + curD))
        Bx = cx + r * math.cos(math.radians(x + y + curD))
        By = cy + r * math.sin(math.radians(x + y + curD))

        Cx = Ax + w * math.cos(math.radians(z + curD))
        Cy = Ay + w * math.sin(math.radians(z + curD))
        Dx = Bx + w * math.cos(math.radians(z + curD))
        Dy = By + w * math.sin(math.radians(z + curD))
        
        cv2.line(temp2, (cx,cy), (int(Ax), int(Ay)), (0,0,255), 1)
        cv2.line(temp2, (cx,cy), (int(Bx), int(By)), (0,255,0), 1)
        cv2.line(temp2, (int(Ax), int(Ay)), (int(Cx), int(Cy)), (0,0,255), 1)
        cv2.line(temp2, (int(Bx), int(By)), (int(Dx), int(Dy)), (0,255,0), 1)
    cv2.imshow("temp2", temp2)

##    manD = cv2.getTrackbarPos(tempMeth, window_name)
##    manD *= 15
##    for deg in range(manD, manD+1, 15):
    start_time= time.time()
    pos_img = test_img.copy()
    for deg in range(0, 180, 5):
        mask_start = time.time()
        if deg > 90:
            offset = 0
        else:
            offset = 1
        curD = (270 + deg + offset)
        if curD >= 360: curD -= 360

        rot_mask = np.zeros((all_mask.shape[0], all_mask.shape[1]), np.uint8)

        Ax = cx + r * math.cos(math.radians(x + curD))
        Ay = cy + r * math.sin(math.radians(x + curD))
        Bx = cx + r * math.cos(math.radians(x + y + curD))
        By = cy + r * math.sin(math.radians(x + y + curD))

        Cx = Ax + w * math.cos(math.radians(z + curD))
        Cy = Ay + w * math.sin(math.radians(z + curD))
        Dx = Bx + w * math.cos(math.radians(z + curD))
        Dy = By + w * math.sin(math.radians(z + curD))

        pts = np.array([[Ax,Ay], [Bx,By], [Cx, Cy], [Dx, Dy]], np.int32)
        rect= cv2.minAreaRect(pts)
        rot = cv2.boxPoints(rect)
        rot = np.int0(rot)
        cv2.drawContours(rot_mask, [rot], 0, (255,255,255), -1)

        curAll = cv2.bitwise_and(all_mask, all_mask, mask=rot_mask)
        curTest = cv2.bitwise_and(test_img, test_img, mask=rot_mask)

##        print(Ax, Ay, Bx, By, Cx, Cy, Dx, Dy)
##        print(rect)
##        print(rot)
##        tx,ty = rot[0]
##        cv2.line(curAll, (cx,cy), (tx,ty), (0,0,255), 1, cv2.LINE_AA)

        b_x,b_y,b_w,b_h = cv2.boundingRect(pts)
##        cv2.rectangle(curAll, (bx,by), (bx+bw,by+bh), (0,255,0), 1)

        cv2.imshow("cur mask", curAll)
        cv2.imshow("test mask", curTest)

##        print(bx,by, bw, bh)
        template = curAll[b_y:b_y+b_h, b_x:b_x+b_w]
        cv2.imshow("template", template)
##        print(template)
##        print()
##        print(curTest[by:by+bh, bx:bx+bw])

        temp_seg = curTest[b_y:b_y+b_h, b_x:b_x+b_w]
        print("masking time:", time.time() - mask_start)

        matching_time = time.time()

        eps = 8
        matchLim = 0.11
        acc = 0
        count = 0
        for outer in range(len(template)):
            for innerx in range(len(template[outer])):
    ##            print(template[0][x])
                for innery in range(len(template[outer][innerx])):
                    if template[outer][innerx][innery] == 0:
                        continue
    ##                print(template[0][x][y], temp_seg[0][x][y])
                    if abs(int(template[outer][innerx][innery]) - int(temp_seg[outer][innerx][innery])) < eps:
                        acc += 1
                    count += 1
        
        cv2.line(pos_img, (cx,cy), (int(Ax), int(Ay)), (0,0,255), 1)
        cv2.line(pos_img, (cx,cy), (int(Bx), int(By)), (0,255,0), 1)
        cv2.imshow("test img", pos_img)

        print("res", acc, count, acc/count)
        if acc/count > matchLim:
            print("at pos")
        else:
            print("not at pos")

        print("matching time:", time.time() - matching_time)
        if acc / count <= matchLim:
            print("pos at", curD-5-offset)
            break
    print("done with", curD)
    print("time:", time.time()-start_time)
    print()

##        match_meth = eval(methods[cv2.getTrackbarPos(tempMeth, window_name)])
##        match_img = curTest.copy()
##        res = cv2.matchTemplate(curTest, template, match_meth)
##        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
##
##        if match_meth in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
##            top_left = min_loc
##        else:
##            top_left = max_loc
##        bottom_right = (top_left[0] + bw, top_left[1] + bh)
##        cv2.rectangle(match_img,top_left, bottom_right, 255, 2)
##
##        cv2.imshow("matched image", match_img)
        
        
cv2.namedWindow(window_name)
cv2.createTrackbar(image_value, window_name, 0, image_val_max, match)
cv2.createTrackbar(tempMeth, window_name, 0, 12, match)

match(0)
cv2.waitKey(0)
