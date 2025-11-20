import cv2

def apply_gaussian_simple(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    filtered = cv2.GaussianBlur(image, (9, 9), 50.0)
    return image, filtered

if __name__ == "__main__":
    image = cv2.imread('va.png', cv2.IMREAD_GRAYSCALE)
    filtered = cv2.GaussianBlur(image, (9, 9), 50.0)
    cv2.imwrite('original_simple.png', image)
    cv2.imwrite('filtered_simple.png', filtered)
    print("Готово!")