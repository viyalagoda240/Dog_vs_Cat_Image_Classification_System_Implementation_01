# 🐶🐱 Dog vs Cat Image Classification System
### COM 3303 – Artificial Intelligence | Rajarata University of Sri Lanka
### Mini Project – Phase 03, Implementation 1

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-87.10%25-brightgreen)
![License](https://img.shields.io/badge/License-Academic-lightgrey)


---

## 📌 Project Overview

A binary image classification system built using a **Convolutional Neural Network (CNN)** that automatically classifies images as either a **Dog** or a **Cat**. The model is trained on 20,000 images and achieves **87.10% accuracy** on a held-out test set of 5,000 images.

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | **87.10%** |
| Test Loss | 0.8547 |
| Training Accuracy | ~98% |
| Validation Accuracy | ~85% |
| Cats Correctly Classified | 2177 / 2500 |
| Dogs Correctly Classified | 2178 / 2500 |

---

## 🗂️ Repository Structure

```
Dog_vs_Cat_Image_Classification_System_Implementation_01/
│
├── dog_cat_classification.ipynb   # Main training notebook
├── model_v1.h5                    # Best model (highest val_accuracy)
├── model_v1_final.h5              # Final epoch model
└── README.md                      # This file
```

---

## 🧠 Model Architecture

The model uses a **3-block CNN** followed by a dense classification head:

```
Input (64×64×3)
     │
     ▼
┌─────────────────────┐
│  Conv2D (32 filters)│  ← Block 1
│  BatchNormalization │
│  MaxPooling2D (2×2) │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Conv2D (64 filters)│  ← Block 2
│  BatchNormalization │
│  MaxPooling2D (2×2) │
└─────────────────────┘
     │
     ▼
┌──────────────────────┐
│ Conv2D (128 filters) │  ← Block 3
│ BatchNormalization   │
│ MaxPooling2D (2×2)   │
└──────────────────────┘
     │
     ▼
  Flatten
     │
     ▼
 Dense (256, ReLU)
     │
 Dropout (0.5)
     │
     ▼
 Dense (1, Sigmoid)   ← Output: 0=Cat, 1=Dog
```

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Loss Function | Binary Cross-Entropy |
| Batch Size | 64 |
| Epochs | 30 |
| Image Size | 64 × 64 pixels |

---

## 📦 Dataset

- **Source:** [Kaggle – Dogs vs Cats](https://www.kaggle.com/datasets/salader/dogsvscats)
- **Total Images:** 25,000
- **Training:** 20,000 (10,000 cats + 10,000 dogs)
- **Test:** 5,000 (2,500 cats + 2,500 dogs)
- **Balance:** Perfectly balanced dataset

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/viyalagoda240/Dog_vs_Cat_Image_Classification_System_Implementation_01.git
cd Dog_vs_Cat_Image_Classification_System_Implementation_01
```

### 2. Install dependencies
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn pillow kaggle
```

### 3. Open the notebook
```bash
jupyter notebook dog_cat_classification.ipynb
```
Or open directly in **Google Colab**.

---

## 🚀 How to Use the Trained Model

### Load and predict with a single image

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the saved model
model = tf.keras.models.load_model("model_v1.h5")

# Load and preprocess your image
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label      = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    print(f"Prediction : {label}")
    print(f"Confidence : {confidence * 100:.2f}%")

# Example usage
predict_image("your_image.jpg")
```

### Expected output
```
Prediction : Dog 🐶
Confidence : 94.32%
```

---

## 📈 Results

### Training & Validation Curves
The model shows clear **overfitting** — training accuracy reaches ~98% while validation accuracy plateaus at ~85%. This will be addressed in Implementation 2.

### Confusion Matrix
```
                Predicted
                Cat    Dog
Actual  Cat  [ 2177    323 ]
        Dog  [  322   2178 ]
```

---

## ⚠️ Known Limitations (Implementation 1)

- **Overfitting:** Training accuracy (98%) significantly higher than test accuracy (87%)
- **No data augmentation:** Model sees identical images each epoch
- **Small image size (64×64):** Fine-grained features are lost
- **No early stopping:** Trains full 30 epochs regardless of improvement
- **Binary only:** Cannot classify animals other than cats and dogs

These will all be fixed in **Implementation 2** using data augmentation, dropout tuning, early stopping, and improved preprocessing.

---

## 🛠️ Tools & Technologies

| Category | Technology |
|----------|-----------|
| Language | Python 3.x |
| Deep Learning | TensorFlow 2.x, Keras |
| Image Processing | Pillow (PIL) |
| Visualization | Matplotlib, Seaborn |
| Evaluation | Scikit-learn |
| Environment | Google Colab / Jupyter Notebook |
| Dataset | Kaggle – Dogs vs Cats |

---

## 📚 References

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
2. Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning Publications.
3. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep CNNs. NIPS.
4. Kaggle. Dogs vs Cats Dataset. https://www.kaggle.com/datasets/salader/dogsvscats
5. Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training. ICML.

---

*Department of Computing | Rajarata University of Sri Lanka | COM 3303 – Artificial Intelligence*
