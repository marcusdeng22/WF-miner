import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from PIL import ImageGrab
'''
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
cv2.imshow("img2 thresh", img2)
##mask = np.zeros(img2.shape, np.uint8)
##img3 = cv2.bitwise_and(img2, mask)
##cv2.imshow("img3", img3)

##print(img.shape)
##print(img2.shape)
##img3 = cv2.bitwise_and(img, img2)
##cv2.imshow("img3", img3)
cv2.waitKey(0)
'''

while (True):
    img = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(655, 305, 955, 605))), cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", img)
    _, img2 = cv2.threshold(img, 245, 255, cv2.THRESH_BINARY)    #230-245
    cv2.imshow("prog", img2)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
