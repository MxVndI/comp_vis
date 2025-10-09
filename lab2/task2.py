import cv2
import numpy as np

video = cv2.VideoCapture(0)
ok, img = video.read()
cv2.namedWindow('Camera', cv2.WINDOW_AUTOSIZE)
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

while (True):
    ok, img = video.read()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    redor_maska = cv2.inRange(hsv, lower_red1, upper_red1)
    redpur_maska = cv2.inRange(hsv, lower_red2, upper_red2)
    red_maska = cv2.bitwise_or(redor_maska, redpur_maska)
    hsv_red_result = cv2.bitwise_and(img, img, mask=red_maska)

    cv2.imshow('Treshold', red_maska)
    cv2.imshow('Camera', hsv_red_result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()