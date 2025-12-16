import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import cv2

# Загрузка CNN-модели
# model = tf.keras.models.load_model("mnist_cnn_model.keras")
model = tf.keras.models.load_model("mnist_model.h5")

root = tk.Tk()
root.title("MNIST-рисовалка (CNN)")

canvas_size = 280

# ЧЁРНЫЙ холст
canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="black")
canvas.pack()

# ЧЁРНОЕ изображение
image = Image.new("L", (canvas_size, canvas_size), 0)
draw = ImageDraw.Draw(image)

def paint(event):
    r = 4
    canvas.create_oval(
        event.x - r, event.y - r,
        event.x + r, event.y + r,
        fill="white", outline="white"
    )
    draw.ellipse(
        [event.x - r, event.y - r,
         event.x + r, event.y + r],
        fill=255
    )

canvas.bind("<B1-Motion>", paint)

def clear():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_size, canvas_size], fill=0)
    label.config(text="")

def predict():
    img = np.array(image)

    # если ничего не нарисовано
    if np.max(img) == 0:
        label.config(text="Нарисуйте цифру")
        return

    # обрезка по цифре
    ys, xs = np.where(img > 0)
    cy = int(np.mean(ys))
    cx = int(np.mean(xs))

    size = max(img.shape)
    square = np.zeros((size, size), dtype=np.uint8)

    y0 = size // 2 - cy
    x0 = size // 2 - cx

    for y, x in zip(ys, xs):
        ny, nx = y + y0, x + x0
        if 0 <= ny < size and 0 <= nx < size:
            square[ny, nx] = img[y, x]

    img = square

    # центрирование
    h, w = img.shape
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.uint8)

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = img

    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.resize(square, (28, 28))
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    prediction = model.predict(img, verbose=0)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)

    label.config(
        text=f"Распознанная цифра: {digit} ({confidence:.1%})"
    )

tk.Button(root, text="Распознать", command=predict).pack()
tk.Button(root, text="Очистить", command=clear).pack()

label = tk.Label(root, text="", font=("Arial", 18))
label.pack()

root.mainloop()
