import numpy as np
import cv2
import os

cap = cv2.VideoCapture('D:\coursera\maskrcnn\\test.mp4')

while (cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

