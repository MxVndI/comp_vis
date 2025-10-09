import cv2
import numpy as np

video = cv2.VideoCapture(0)
ok, img = video.read()

low_redor = np.array([0, 120, 70])
up_redor = np.array([10, 255, 255])
low_redpur = np.array([170, 120, 70])
up_redpur = np.array([180, 255, 255])

kernel = np.ones((5, 5), np.uint8)

while (True):
    ok, img = video.read()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # COLOR_BGR2RGB, COLOR_BGR2RGB
    redor_maska = cv2.inRange(hsv, low_redor, up_redor)
    redpur_maska = cv2.inRange(hsv, low_redpur, up_redpur)
    red_maska = cv2.bitwise_or(redor_maska, redpur_maska)
    hsv_red_result = cv2.bitwise_and(img, img, mask=red_maska)

    opening_img = cv2.morphologyEx(hsv_red_result, cv2.MORPH_OPEN, kernel)

    closing_img = cv2.morphologyEx(hsv_red_result, cv2.MORPH_CLOSE, kernel)

    dilated_img = cv2.dilate(hsv_red_result, kernel)
    eroded_img = cv2.erode(hsv_red_result, kernel)

    cv2.imshow('Close', closing_img)
    cv2.imshow('Open', opening_img)
    cv2.imshow('Dila', dilated_img)
    cv2.imshow('Ero', eroded_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()