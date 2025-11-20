import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))
    center = size // 2

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    kernel = kernel / np.sum(kernel)
    return kernel


def apply_gaussian_filter(image, kernel):
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2

    padded_image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    filtered_image = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i + kernel_size, j:j + kernel_size]
            filtered_image[i, j] = np.sum(window * kernel)

    return filtered_image.astype(np.uint8)


def main_custom():
    image = cv2.imread('va.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel_size = 9
    sigma = 50.0
    kernel = create_gaussian_kernel(kernel_size, sigma)

    filtered_image = apply_gaussian_filter(gray_image, kernel)

    cv2.imwrite('original_custom.png', gray_image)
    cv2.imwrite('filtered_custom.png', filtered_image)

if __name__ == "__main__":
    main_custom()