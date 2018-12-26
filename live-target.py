import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from PIL import ImageGrab

while (True):   #perhaps add a sleep?
    img = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(655, 305, 955, 605))), cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", img)
    _, img2 = cv2.threshold(img, 245, 255, cv2.THRESH_BINARY)    #230-245
    cv2.imshow("prog", img2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
