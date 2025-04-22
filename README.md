#  Anemia Detection Using Nail Beds with CNN, SMOTE, and Cross-Validation

A deep learning-based diagnostic system that detects anemia from fingernail bed images using Convolutional Neural Networks (CNNs). The model addresses class imbalance with SMOTE and ensures robustness using k-fold cross-validation.

---


##  Features

- Image-based Anemia Classification (Anemic / Not Anemic)
- CNN Model (MobileNetV2)
- Class Imbalance handled using **SMOTE**
- Evaluation with **k-Fold Cross-Validation**
- Flask-based Web API for predictions
- Easy image upload and JSON output
- Scalable and mobile-ready backend

---

##  Tech Stack

| Component           | Tool/Library Used           |
|---------------------|-----------------------------|
| Deep Learning       | TensorFlow, Keras           |
| Image Processing    | OpenCV, Pillow              |
| Backend API         | Flask                       |
| Frontend (optional) | Streamlit (alternative GUI) |
| Data Handling       | NumPy, Pandas               |
| Visualization       | Matplotlib, Seaborn         |

---

## 📂 Project Structure
├── app.py                 # Flask API
├── models/
│   └── fingernail_anemia_mobilenetv2.keras
├── static/
│   └── (Optional image/static files)
├── templates/
│   └── index.html         # Web interface
├── requirements.txt
└── README.md
