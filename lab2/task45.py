import cv2
import numpy as np

video = cv2.VideoCapture(0)
ok, path = video.read()

path = np.zeros_like(path)
low_redor = np.array([0, 120, 70])
up_redor = np.array([10, 255, 255])
low_redpur = np.array([170, 120, 70])
up_redpur = np.array([180, 255, 255])

lastx = 0
lasty = 0
first_frame = True
print("Моменты первого порядка:")

while True:
    ok, img = video.read()
    if not ok:
        break

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    redor_maska = cv2.inRange(hsv, low_redor, up_redor)
    redpur_maska = cv2.inRange(hsv, low_redpur, up_redpur)
    red_maska = cv2.bitwise_or(redor_maska, redpur_maska)
    hsv_red_result = cv2.bitwise_and(img, img, mask=red_maska)

    kernel = np.ones((5, 5), np.uint8)
    red_maska = cv2.morphologyEx(red_maska, cv2.MORPH_CLOSE, kernel)
    red_maska = cv2.morphologyEx(red_maska, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(red_maska, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    center_x = lastx
    center_y = lasty
    object_found = False

    for i, contour in enumerate(contours):
        first_moments = cv2.moments(contour)

        square = first_moments['m00']
        if square < 500:
            continue

        object_found = True
        summ_x = first_moments['m10']
        summ_y = first_moments['m01']

        print(f"Контур {i}: Площадь: {square:.0f}, Сумма по x: {summ_x:.0f}, Сумма по y: {summ_y:.0f}")

        if square > 0:
            center_x = int(summ_x / square)
            center_y = int(summ_y / square)
        else:
            center_x, center_y = lastx, lasty

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 3)

        cv2.circle(img, (center_x, center_y), 5, (123, 0, 255), -1)

        cv2.putText(img, f"({center_x}, {center_y})",
                    (center_x + 10, center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(img, f"Area: {square:.0f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if object_found:
        if not first_frame:
            cv2.line(path, (lastx, lasty), (center_x, center_y), (90, 0, 0), 5)

        lastx = center_x
        lasty = center_y
        first_frame = False

    img_with_path = cv2.add(img, path)

    cv2.putText(img_with_path, f"Contours found: {len(contours)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Camera1 - Tracking', img_with_path)
    cv2.imshow('Camera2 - Red Filter', hsv_red_result)
    cv2.imshow('Camera3 - Mask', red_maska)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()