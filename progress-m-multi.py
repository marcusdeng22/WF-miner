import cv2
import math
import numpy as np
import glob
import time
import threading

class retThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return

all_mask = cv2.imread("./filled/20181226213622_1.jpg")[305:605, 655:955]
cv2.imshow("mask orig", all_mask)

window_name = 'Progress Template'

allIm = []
count = 0
for img in glob.glob("./new-train2/*.jpg"):
##for img in glob.glob("./double-train/*.jpg"):
    count += 1
    n = cv2.imread(img)[305:605, 655: 955]
    allIm.append(n)
    if count > 50: break
orig = allIm[0]
image_value = "image"
image_val_max = len(allIm) - 1

w = 18
h = 3.2
r = 117
cx = 144
cy = 145
y = 2 * math.degrees(math.asin((h/2)/r))
x = (5 - y) / 2
z = (y / 2) + x
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

##    print(rot_mask)
    allMasks.append(rot_mask)

    mainMask.append(cv2.bitwise_and(all_mask, all_mask, mask=rot_mask))

    b_x,b_y,b_w,b_h = cv2.boundingRect(pts)
    boundingPts.append([b_x,b_y,b_w,b_h])

##print(allMasks)
##print(mainMask)

eps = 8
matchLim = 0.11
def innerComp(template, segment):
    acc = 0
    count = 0
    for innerx in range(len(template)):
        for innery in range(len(template[innerx])):
            if template[innerx][innery] == 0:
                continue
            if abs(int(template[innerx][innery]) - int(segment[innerx][innery])) < eps:
                acc += 1
            count += 1
    return acc, count

def match(val):
    image_num = cv2.getTrackbarPos(image_value, window_name)
    test_img = allIm[image_num]
    cv2.imshow("test img", test_img)
    
    
##    rot_mask = np.zeros((all_mask.shape[0], all_mask.shape[1]), np.uint8)

    #alternate: start at upper middle and compute the mask on both images and compare
    #then once hit, compute masks simultaneously and compare (do a mask and compare in same iter, rather than after)

    start_time= time.time()
    total_mask_time = 0
    total_matching_time = 0
    total_test_time = 0
    total_test2_time = 0
    failedComp = False
    pos_img = test_img.copy()
    for deg in range(0, 180, 5):
        mask_start = time.time()

        if deg > 90:
            offset = 0
        else:
            offset = 1
        curD = (270 + deg + offset)
        if curD >= 360: curD -= 360
        
##        curAll = cv2.bitwise_and(all_mask, all_mask, mask=rot_mask)
        curAll = mainMask[int(deg/5)]
##        curTest = cv2.bitwise_and(test_img, test_img, mask=rot_mask)
        
        curTest = cv2.bitwise_and(test_img, test_img, mask=allMasks[int(deg/5)])

##        b_x,b_y,b_w,b_h = cv2.boundingRect(pts)
        b_x,b_y,b_w,b_h = boundingPts[int(deg/5)]

        #this takes most of the time: limit the number of displays!
##        cv2.imshow("cur mask", curAll)
##        cv2.imshow("test mask", curTest)

        template = curAll[b_y:b_y+b_h, b_x:b_x+b_w]
##        cv2.imshow("template", template)
        temp_seg = curTest[b_y:b_y+b_h, b_x:b_x+b_w]
        
        
###############################################################################################################################################
        
        print("masking time:", time.time() - mask_start)
        total_mask_time += time.time() - mask_start

        matching_time = time.time()

        #replace below with a multithreaded version?- 1 on each subarray of template
        #replaced with numpy solution
##        acc = 0
##        count = 0
##        for outer in range(len(template)):
##            for innerx in range(len(template[outer])):
##                for innery in range(len(template[outer][innerx])):
##                    if template[outer][innerx][innery] == 0:
##                        continue
##                    if abs(int(template[outer][innerx][innery]) - int(temp_seg[outer][innerx][innery])) < eps:
##                        acc += 1
##                    count += 1
##        print("matching time:", time.time() - matching_time)
##        print(len(template), len(template[0]))
##        total_matching_time += time.time() - matching_time
##        
##        #multithread
##        test_start = time.time()
##        threads = []
##        for templateAr, segAr in zip(template, temp_seg):
##            t1 = retThread(target=innerComp, args=(templateAr, segAr,))
##            t1.start()
##            threads.append(t1)
##
##        acc2 = 0
##        count2 = 0        
##        for t in threads:
##            t_acc, t_count = t.join()
##            acc2 += t_acc
##            count2 += t_count
##        total_test_time += time.time() - test_start

        #numpy sol
        test2_start = time.time()
        tempArr = template.flatten()
        idx = tempArr.nonzero()
        templateT = tempArr[idx].astype(np.int16)
        segT = temp_seg.flatten()[idx].astype(np.int16)
        diff = np.absolute(templateT - segT)
        count = len(diff)
        acc = len(np.where(diff < eps)[0])
        total_test2_time += time.time() - test2_start

##        if acc2 != acc or count2 != count or acc3 != acc or count3 != count:
##            print("difference in matching!", acc, count, acc2, count2, acc3, count3)
##            failedComp = True
        
        x = round(cx + math.degrees(math.cos(math.radians(curD + 2.5))) * 200)
        y = round(cy + math.degrees(math.sin(math.radians(curD + 2.5))) * 200)
        cv2.line(pos_img, (cx, cy), (x,y), (255,0,0), 1, cv2.LINE_AA)
        cv2.imshow("test img", pos_img)

##        print("res", acc, count, acc/count)
##        if acc/count > matchLim:
##            print("at pos")
##        else:
##            print("not at pos")

        
        if acc / count <= matchLim:
            print("pos at", curD-5-offset)
            break
    print("done with", curD)
    print("total time:", time.time()-start_time)
    print("total masking time: ", total_mask_time)
    print("total matching time:", total_matching_time)
    print("total test time:", total_test_time)
    print("total test2 time:", total_test2_time)
    print("failed?", failedComp)
    print()
        
        
cv2.namedWindow(window_name)
cv2.createTrackbar(image_value, window_name, 0, image_val_max, match)

match(0)
cv2.waitKey(0)
