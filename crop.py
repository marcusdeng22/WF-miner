import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

##orig = cv2.imread("./train/20181121013009_1.jpg")[305:605, 655:955] #in prog early
orig = cv2.imread("./train/20181121013019_1.jpg")[305:605, 655:955] #in prog after
##orig = cv2.imread("./train/20181121013018_1.jpg")[305:605, 655:955] #double
##orig = cv2.imread("./train/20181121015035_1.jpg")[305:605, 655:955] #overlap targets
##orig = cv2.imread("./train/20181121015221_1.jpg")[305:605, 655:955] #mixed back
##orig = cv2.imread("./train/20181121015223_1.jpg")[305:605, 655:955] #at end
##orig = cv2.imread("./train/20181121015422_1.jpg")[305:605, 655:955] #overlap with hit
##orig = cv2.imread("./train/20181121015443_1.jpg")[305:605, 655:955] #overlap with done (top is vis)
##orig = cv2.imread("./train/20181121015447_1.jpg")[305:605, 655:955] #overlap with done
##orig = cv2.imread("./train/20181121122156_1.jpg")[305:605, 655:955] #stalk overlap
##orig = cv2.imread("./train/20181121012950_1.jpg")[305:605, 655:955] #perfect hit

mask_c = np.zeros((orig.shape[0], orig.shape[1]), np.uint8)
cv2.circle(mask_c, (145, 145), 125, (255,255,255), thickness=21)
print(mask_c)
##cv2.circle(mask_c, (145, 145), 138, (255,255,255), thickness=-1)
##cv2.circle(mask_i, (145, 145), 114, (255,255,255), thickness=-1)

mask_img = cv2.bitwise_and(orig, orig, mask=mask_c)

cv2.imshow("mask", mask_img)

##pts = np.array([[100,100], [200,150], [300, 125], [225, 75]], np.int32)

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
'''
print(cx + r * math.cos(math.radians(x)))
print(cy + r * math.cos(math.radians(y)))

Ax = cx + r * math.cos(math.radians(x))
Ay = cy + r * math.sin(math.radians(y))
Bx = cx + r * math.cos(math.radians(x + y))
By = cy + r * math.sin(math.radians(x + y))

Cx = Ax + w * math.cos(math.radians(z))
Cy = Ay + w * math.sin(math.radians(z))
Dx = Bx + w * math.cos(math.radians(z))
Dy = Ay + w * math.sin(math.radians(z))
'''

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
    if deg == 0: print(rot)
    cv2.drawContours(rot_mask, [rot], 0, (255,255,255), -1)
'''
pts = np.array([[Ax,Ay], [Bx,By], [Cx, Cy], [Dx, Dy]], np.int32)
print(pts)
rect = cv2.minAreaRect(pts)
rot = cv2.boxPoints(rect)
rot = np.int0(rot)
cv2.drawContours(orig, [rot], 0, (0,0,255), 2)
'''
mask_rot_img = cv2.bitwise_and(orig, orig, mask=rot_mask)
cv2.imshow("mask_rot", mask_rot_img)

cv2.waitKey(0)
