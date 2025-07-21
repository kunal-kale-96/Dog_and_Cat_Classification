# 🐶🐱 Dog vs Cat Image Classifier (with GUI)

A simple and interactive image classifier that uses **Transfer Learning** with **MobileNetV2** to differentiate between dogs and cats. The model is trained using TensorFlow, and a user-friendly GUI is built with Tkinter.

---

## 🛠️ Libraries Used

| Library | Purpose |
|--------------|-------------------------------------------------------------------------|
| tensorflow | Core deep learning library for training and using the CNN model |
| numpy | Image array processing and manipulation |
| pillow | Image loading and preprocessing for both training and GUI |
| matplotlib | (Optional) Useful for visualizations in extended versions |
| tkinter | GUI development for easy image classification |

---

## 🚀 How to Run the Project & Provide Input

### 🔧 1. Setup and Installation

1. Make sure Python 3.7+ is installed.
2. Clone this repository or download the ZIP.
3. Open the terminal in the project folder and install the dependencies:

Step 1: Create and Activate a Virtual Environment

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

```bash
pip install -r requirements.txt
```

### 📁 2. Prepare the Dataset

Organize your images like this:

```
dataset/
└── train/
    ├── cat/
    │   ├── cat1.jpg
    │   └── ...
    └── dog/
        ├── dog1.jpg
        └── ...
```

Include **at least 5 images per class** (cat and dog) for demonstration purposes.

### 🧠 3. Train the Model (`train.py`)

This script loads MobileNetV2, applies transfer learning, augments data, and trains the model.

To run:

```bash
python train.py
```

- The model will be trained for 10 epochs.
- After training, the model is saved at: `model/dog_cat_model.keras`.

### 🖼️ 4. Run the GUI (`gui_app.py`)

This script launches a user interface to classify images using the trained model.

To launch:

```bash
python gui_app.py
```

### 🧪 5. Using the Application

A window will open where you can:

- Click **"Choose Image"**
- Select **any image** of a **dog or a cat**
- View the **classification result** displayed on screen with a **confidence score**

Example results:
- 🐶 Dog (87.5%)
- 🐱 Cat (91.2%)

---

## 📥 How to Provide Input

- Use the GUI to select a **JPG**, **PNG**, **BMP**, **GIF**, or **WEBP** file.
- The image will be resized and normalized automatically.
- The model will classify it in real-time.
- The result will appear with the prediction and confidence score.

---

## 📌 Notes

- Only 10 images are used for demonstration. For better accuracy, increase dataset size.
- GUI is styled with dark theme and responsive layout.
- Transfer learning provides high accuracy even with fewer training samples.
- The model uses MobileNetV2 as the base model for efficient performance.

---

## 📂 Project Structure

```
project/
├── dataset/
│   └── train/
│       ├── cat/
│       └── dog/
├── model/
│   └── dog_cat_model.keras
├── train.py
├── gui_app.py
├── requirements.txt
└── README.md
```

---

## 🎯 Features

- **Transfer Learning**: Uses pre-trained MobileNetV2 for efficient training
- **Data Augmentation**: Improves model robustness with image transformations
- **Interactive GUI**: User-friendly interface for real-time classification
- **Multiple Formats**: Supports various image formats (JPG, PNG, BMP, GIF, WEBP)
- **Confidence Scoring**: Shows prediction confidence percentage
- **Dark Theme**: Modern and easy-on-the-eyes interface

---

## 🔧 Requirements

Create a `requirements.txt` file with:

```
tensorflow>=2.8.0
numpy>=1.21.0
pillow>=8.3.0
matplotlib>=3.5.0
```

---
