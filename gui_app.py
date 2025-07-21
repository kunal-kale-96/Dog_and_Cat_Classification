import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import os

MODEL_PATH = "model/dog_cat_model.keras"
IMG_SIZE = (128, 128)

if not os.path.exists(MODEL_PATH):
    messagebox.showerror("Model Missing", f"Model file not found at {MODEL_PATH}")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)

def classify_image(img_path):
    try:
        img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, *IMG_SIZE, 3)
        prediction = model.predict(img_array)[0][0]
        confidence = round(prediction * 100 if prediction > 0.5 else (1 - prediction) * 100, 2)
        if prediction > 0.5:
            return f"🐶 Dog ({confidence}%)", "green"
        else:
            return f"🐱 Cat ({confidence}%)", "blue"
    except Exception as e:
        return f"Error: {str(e)}", "red"

def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.webp")])
    if file_path:
        result, color = classify_image(file_path)
        result_label.config(text=result, fg="white", bg=color)
        img = Image.open(file_path).convert("RGB")
        img.thumbnail((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

root = tk.Tk()
root.title("🐾 Dog vs Cat Classifier")
root.geometry("500x600")
root.configure(bg="#2d2d2d")

title = tk.Label(root, text="🐶🐱 Dog vs Cat Classifier", font=("Helvetica", 18, "bold"), fg="white", bg="#2d2d2d")
title.pack(pady=20)

image_label = tk.Label(root, bg="#2d2d2d")
image_label.pack(pady=10)

choose_btn = tk.Button(root, text="Choose Image", command=browse_image,
                       bg="#ff9800", fg="white", font=("Arial", 14, "bold"),
                       activebackground="#e69100", padx=20, pady=10)
choose_btn.pack(pady=20)

result_label = tk.Label(root, text="", font=("Arial", 16, "bold"), bg="#2d2d2d")
result_label.pack(pady=20)

footer = tk.Label(root, text="Made with ❤️ using Transfer Learning", font=("Arial", 10), fg="gray", bg="#2d2d2d")
footer.pack(side="bottom", pady=10)

root.mainloop()
