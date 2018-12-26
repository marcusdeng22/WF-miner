import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

d_red = (150, 55, 65)
l_red = (250, 200, 200)

orig = cv2.imread("./train/20181121013009_1.jpg")[300:600, 800:955] #in prog early
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

cv2.imshow("color", cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
