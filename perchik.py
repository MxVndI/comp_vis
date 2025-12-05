import cv2
import numpy as np

video = cv2.VideoCapture(0)
ok, img = video.read()
cv2.namedWindow('Camera', cv2.WINDOW_AUTOSIZE)


while (True):
    ok, img = video.read()

    salt_mask_white = np.random.random(img.shape[:2]) < 0.1
    img[salt_mask_white] = 255
    salt_mask_black = np.random.random(img.shape[:2]) < 0.1
    img[salt_mask_black] = 0
    cv2.imshow('Camera', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()