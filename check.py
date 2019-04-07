import numpy as np
import os
import pickle
import cv2

for file in os.listdir("./tmp"):
    img = cv2.imread("./tmp/" + file, 0)
    cv2.imwrite("./tmp/gray_" + file, img)

