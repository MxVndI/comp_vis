import cv2
import numpy as np
import matplotlib.pyplot as plt

from ready_gaus import apply_gaussian_simple
from gausmy import create_gaussian_kernel, apply_gaussian_filter


def compare_gaussian_methods():
    image = cv2.imread('../va.png', cv2.IMREAD_GRAYSCALE)

    print(f"Размер изображения: {image.shape}")

    filtered_opencv = cv2.GaussianBlur(image, (9, 9), 50.0)
    kernel = create_gaussian_kernel(9, 50.0)
    filtered_custom = apply_gaussian_filter(image, kernel)

    cv2.imwrite('original.png', image)
    cv2.imwrite('filtered_opencv.png', filtered_opencv)
    cv2.imwrite('filtered_custom.png', filtered_custom)

    plt.figure(figsize=(16, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(filtered_opencv, cmap='gray')
    plt.title('OpenCV GaussianBlur\n(9x9, σ=50.0)')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(filtered_custom, cmap='gray')
    plt.title('Наша реализация\n(9x9, σ=50.0)')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('comparison_result.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    compare_gaussian_methods()