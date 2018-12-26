import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

d_red = (150, 55, 65)
l_red = (250, 200, 200)

orig = cv2.imread("./train/20181121013009_1.jpg")[300:600, 655:955] #in prog early
##orig = cv2.imread("./train/20181121013019_1.jpg")[300:600, 655:955] #in prog after
##orig = cv2.imread("./train/20181121013018_1.jpg")[300:600, 655:955] #double
##orig = cv2.imread("./train/20181121015035_1.jpg")[300:600, 655:955] #overlap targets
##orig = cv2.imread("./train/20181121015221_1.jpg")[300:600, 655:955] #mixed back
##orig = cv2.imread("./train/20181121015223_1.jpg")[305:605, 655:955] #at end
##orig = cv2.imread("./train/20181121015422_1.jpg")[305:605, 655:955] #overlap with hit
##orig = cv2.imread("./train/20181121015443_1.jpg")[305:605, 655:955] #overlap with done (top is vis)
##orig = cv2.imread("./train/20181121015447_1.jpg")[305:605, 655:955] #overlap with done
##orig = cv2.imread("./train/20181121122156_1.jpg")[305:605, 655:955] #stalk overlap
##orig = cv2.imread("./train/20181121012950_1.jpg")[300:600, 655:955] #perfect hit
img = orig.copy()
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("img", img)
cv2.imshow("img2", img2)

#get progress
ret, prog = cv2.threshold(img2, 90, 255, cv2.THRESH_TOZERO)     #80
ret, prog = cv2.threshold(prog, 150, 0, cv2.THRESH_TOZERO_INV)
##ret, prog = cv2.threshold(prog, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
##plt.hist(img2.ravel(), 256)
##plt.show()
##ret, prog = cv2.threshold(prog, 100, 255, cv2.THRESH_BINARY_INV)

##ret, prog = cv2.threshold(img2, 150, 255, cv2.THRESH_TRUNC)
##ret, prog = cv2.threshold(prog, 140, 255, cv2.THRESH_BINARY)    #130-140; trunc is too variable
##ret, prog = cv2.threshold(img2, 80, 255, cv2.THRESH_BINARY)
cv2.imshow("prog", prog)


#this gets the target
##mask = np.ones(img2.shape, np.uint8)
ret, img2 = cv2.threshold(img2, 245, 255, cv2.THRESH_BINARY)    #230-245

#dilate
kernel1 = np.ones((1,1), np.uint8)
kernel2 = np.ones((2,2), np.uint8)
img2 = cv2.erode(img2, kernel1, iterations=10)
cv2.imshow("img2 thresh", img2)
#detect corners on the target

dst = cv2.cornerHarris(img2,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
cv2.imshow("corners", img)



##mask = np.zeros(img2.shape, np.uint8)
##img3 = cv2.bitwise_and(img2, mask)
##cv2.imshow("img3", img3)

##print(img.shape)
##print(img2.shape)
##img3 = cv2.bitwise_and(img, img2)
##cv2.imshow("img3", img3)
cv2.waitKey(0)

'''
##detector = cv2.FeatureDetector_create('MSER')
detector = cv2.MSER_create()
fs = detector.detect(img2)
fs.sort(key = lambda x: -x.size)

def supress(x):
        for f in fs:
                distx = f.pt[0] - x.pt[0]
                disty = f.pt[1] - x.pt[1]
                dist = math.sqrt(distx*distx + disty*disty)
                if (f.size > x.size) and (dist<f.size/2):
                        return True

sfs = [x for x in fs if not supress(x)]

for f in sfs:
        cv2.circle(img, (int(f.pt[0]), int(f.pt[1])), int(f.size/2), d_red, 2)
        cv2.circle(img, (int(f.pt[0]), int(f.pt[1])), int(f.size/2), l_red, 1)

h, w = orig.shape[:2]
vis = np.zeros((h, w*2+5), np.uint8)
vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
vis[:h, :w] = orig
vis[:h, w+5:w*2+5] = img

cv2.imshow("image", vis)
cv2.waitKey(0)
'''
